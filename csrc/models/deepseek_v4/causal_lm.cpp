#include "compressor.hpp"
#include "hyper_connection.hpp"
#include "model_config.hpp"
#include "native_ops.hpp"
#include "request_rows.hpp"
#include "rotary_frequencies.hpp"

#include "engine/deepseek_v4_metadata.hpp"
#include "engine/rank_barrier.hpp"
#include "engine/distributed/communication_group.hpp"
#include "execution/submit.hpp"
#include "global_state/forward_context.hpp"
#include "layers/embedding/embedding.hpp"
#include "layers/linear/dense.hpp"
#include "models/models_registry.hpp"

#include <algorithm>
#include <cstdio>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <string_view>
#include <stdexcept>
#include <utility>

namespace infinilm::models::deepseek_v4 {
namespace {

using runtime::DataType;
using runtime::Runtime;
using runtime::Shape;
using runtime::Tensor;
using weights::Weight;
using C4SequentialIndex = infini::ops::DeepseekV4C4IndexSequential;
using ModelConfig = config::ModelConfig;
using Rank = engine::distributed::RankInfo;
using Linear = nn::Linear;
using ColumnLinear = nn::ColumnParallelLinear;
using RowLinear = nn::RowParallelLinear;
using RMSNorm = layers::RMSNorm;
using Quant = quantization::CompressedTensors;

constexpr size_t kV4TensorParallelSize = 8;

struct V4Config {
    size_t hidden, vocab, heads, output_groups, q_rank, o_rank, experts, intermediate;
    size_t topk, hash_layers, index_heads, streams, iterations, max_positions;
    double rms_eps, hc_eps, route_scale, swiglu_limit;
    std::vector<size_t> ratios;
    std::shared_ptr<Quant> quant;
    bool is_mtp, classic_mtp_target;

    V4Config(const ModelConfig &config, const Rank &rank)
        : hidden(config.get<size_t>("hidden_size")), vocab(config.get<size_t>("vocab_size")),
          heads(config.get<size_t>("num_attention_heads")), output_groups(config.get<size_t>("o_groups")),
          q_rank(config.get<size_t>("q_lora_rank")), o_rank(config.get<size_t>("o_lora_rank")),
          experts(config.get<size_t>("n_routed_experts")), intermediate(config.get<size_t>("moe_intermediate_size")),
          topk(config.get<size_t>("num_experts_per_tok")), hash_layers(config.get<size_t>("num_hash_layers")),
          index_heads(config.get<size_t>("index_n_heads")), streams(config.get<size_t>("hc_mult")),
          iterations(config.get<size_t>("hc_sinkhorn_iters")), max_positions(config.get<size_t>("max_position_embeddings")),
          rms_eps(config.get<double>("rms_norm_eps")), hc_eps(config.get<double>("hc_eps")),
          route_scale(config.get<double>("routed_scaling_factor")), swiglu_limit(config.get<double>("swiglu_limit")),
          ratios(config.get<std::vector<size_t>>("compress_ratios")),
          is_mtp(config.get<std::string>("model_type") == "deepseek_v4_mtp"),
          classic_mtp_target(!is_mtp && config.get_or<bool>(kClassicMtpTargetKey, false)) {
        const auto layers = config.get<size_t>("num_hidden_layers");
        const auto &json = config.get_config_json();
        if (!json.contains("quantization_config") || !json.at("quantization_config").is_object() ||
            json.at("quantization_config").value("quant_method", "") != std::string("compressed-tensors")) {
            throw std::invalid_argument("DeepSeek V4 mainline currently requires the channel INT8 W8A8 checkpoint");
        }
        if (rank.device.type() != runtime::Device::Type::kHygon || rank.tp_size != kV4TensorParallelSize || rank.pp_size != 1 || rank.tp_rank < 0 || rank.tp_rank >= rank.tp_size || !rank.comm || config.get_dtype() != DataType::kBFloat16 || config.get<size_t>("head_dim") != 512 || config.get<size_t>("qk_rope_head_dim") != 64 || config.get<size_t>("num_key_value_heads") != 1 || config.get<size_t>("n_shared_experts") != 1 || config.get<size_t>("index_head_dim") != 128 || config.get<size_t>("index_topk") != 512 || config.get<size_t>("sliding_window") != 128 || config.get<std::string>("scoring_func") != "sqrtsoftplus" || !config.get_or<bool>("norm_topk_prob", true) || !heads || heads % 8 || output_groups != 8 || !intermediate || intermediate % 8 || !hidden || !vocab || !q_rank || !o_rank || !experts || !topk || topk > experts || !index_heads || streams != 4 || !iterations || !max_positions || !layers || ratios.size() < layers || (is_mtp && (layers != 1 || hash_layers != 0 || ratios != std::vector<size_t>{0})) || hash_layers > layers || !std::isfinite(swiglu_limit) || swiglu_limit <= 0) {
            throw std::invalid_argument("Unsupported DeepSeek V4 Flash W8A8 configuration");
        }
        ratios.resize(layers);
        quant = std::make_shared<Quant>(json.at("quantization_config"));
    }
};

template<class Op, class... Args>
void submit(Runtime &rt, const Args &...args) {
    execution::submit<Op>(rt, Op::DefaultConfig(rt.device().type()), args...);
}

Tensor add(Runtime &rt, const Tensor &a, const Tensor &b) {
    auto out = Tensor::empty(rt, a->shape(), a->dtype());
    submit<infini::ops::Add>(rt, a, b, out);
    return out;
}

Tensor cast(Runtime &rt, const Tensor &value, DataType dtype) {
    auto out = Tensor::empty(rt, value->shape(), dtype);
    submit<infini::ops::Cast>(rt, value, out);
    return out;
}

// The HCU custom reduction used by the reference accumulates rank-local
// BF16 contributions in FP32 and truncates only the final writeback.
Tensor reduce_tp(Runtime &rt, const Tensor &local, infinicclComm_t comm,
                 const std::vector<size_t> &query_offsets, bool packed_collectives) {
    auto sum = cast(rt,local,DataType::kFloat32);
    if (packed_collectives) {
        engine::distributed::allreduce_(rt, sum, sum, infinicclSum, comm);
    } else {
        for_each_request(query_offsets, [&](const RequestRows &rows) {
            auto part = rows.slice(sum);
            engine::distributed::allreduce_(rt, part, part, infinicclSum, comm);
        });
    }
    auto output = Tensor::empty(rt,local->shape(),local->dtype());
    submit<infini::ops::DeepseekV4CastRz>(rt,sum,output);
    return output;
}

Tensor normalize_query_and_rotate(Runtime &rt, const Tensor &input, const Tensor &weight,
                                  float epsilon, const Tensor &positions, const Tensor &frequencies) {
    if (rt.device().type() == runtime::Device::Type::kHygon && input->dtype() == DataType::kBFloat16) {
        auto output = Tensor::empty(rt, input->shape(), input->dtype());
        submit<infini::ops::DeepseekV4QueryNormRope>(rt, input, weight, positions,
                                                   frequencies, epsilon, output);
        return output;
    }
    auto fp32 = cast(rt, input, DataType::kFloat32);
    auto normalized = Tensor::empty(rt, input->shape(), DataType::kFloat32);
    submit<infini::ops::RmsNorm>(rt, fp32, weight, epsilon, normalized);
    auto rounded = cast(rt, normalized, input->dtype());
    apply_rotary_tail(rt, rounded, positions, frequencies);
    return rounded;
}

struct MixWeights {
    Tensor projection, scale, base;
};

struct C4ScorePlan {
    static constexpr size_t ranks = 8;
    std::array<std::vector<RequestRows>, ranks> scored;
    std::vector<RequestRows> sequential;
    std::array<size_t, ranks> row_counts{};
    size_t gathered_rows{};
};

std::optional<C4ScorePlan> make_c4_score_plan(const int32_t *lengths, size_t rows) {
    constexpr auto top = static_cast<int32_t>(C4SequentialIndex::kMaxCandidates);
    constexpr size_t stripe_rows = 512;
    constexpr size_t wide_score_min_rows = 1024;
    struct Stripe {
        RequestRows rows;
        uint64_t candidates;
    };

    const auto contiguous_rows = (rows + C4ScorePlan::ranks - 1) / C4ScorePlan::ranks;
    if (contiguous_rows < wide_score_min_rows) {
        return std::nullopt;
    }

    C4ScorePlan plan;
    std::vector<Stripe> stripes;
    std::array<uint64_t, C4ScorePlan::ranks> contiguous_work{};
    for (size_t first = 0; first < rows;) {
        const bool scored = lengths[first] > top;
        size_t end = first + 1;
        while (end < rows && (lengths[end] > top) == scored
               && (!scored || end - first < stripe_rows)) {
            ++end;
        }
        const RequestRows span{first, end - first};
        if (scored) {
            uint64_t candidates = 0;
            for (size_t row = first; row < end; ++row) {
                const auto length = static_cast<uint64_t>(lengths[row]);
                candidates += length;
                contiguous_work[row / contiguous_rows] += length;
            }
            stripes.push_back({span, candidates});
        } else {
            plan.sequential.push_back(span);
        }
        first = end;
    }

    std::stable_sort(stripes.begin(), stripes.end(), [](const Stripe &a, const Stripe &b) {
        return a.candidates > b.candidates;
    });
    std::array<uint64_t, C4ScorePlan::ranks> work{};
    for (const auto &stripe : stripes) {
        const auto rank = std::min_element(work.begin(), work.end()) - work.begin();
        plan.scored[rank].push_back(stripe.rows);
        plan.row_counts[rank] += stripe.rows.count;
        work[rank] += stripe.candidates;
    }
    plan.gathered_rows = *std::max_element(plan.row_counts.begin(), plan.row_counts.end());
    const auto least_rows = *std::min_element(plan.row_counts.begin(), plan.row_counts.end());
    const auto previous_work = *std::max_element(contiguous_work.begin(), contiguous_work.end());
    const auto balanced_work = *std::max_element(work.begin(), work.end());
    // Keep the wide score kernel on every rank. Require a smaller gather
    // and a meaningful decrease in the maximum per-rank candidate count.
    if (plan.sequential.empty() || least_rows < wide_score_min_rows
        || plan.gathered_rows >= contiguous_rows || balanced_work + balanced_work / 10 >= previous_work) {
        return std::nullopt;
    }

    for (auto &rank : plan.scored) {
        std::sort(rank.begin(), rank.end(), [](const RequestRows &a, const RequestRows &b) {
            return a.first < b.first;
        });
        size_t merged = 0;
        for (const auto span : rank) {
            if (merged && rank[merged - 1].first + rank[merged - 1].count == span.first) {
                rank[merged - 1].count += span.count;
            } else {
                rank[merged++] = span;
            }
        }
        rank.resize(merged);
    }
    return plan;
}

// Three short verify spans can contribute up to four rows apiece.
constexpr size_t kPackedSharedExpertRows = 3 * 4;
constexpr size_t kC4IndexWidth = 128;
constexpr size_t kC4IndexTopK = 512;
constexpr size_t kC4IndexMaxUnscoredPages = kC4IndexTopK / (kV4PageSize / 4);

bool env_enabled(const char *name) {
    const auto *mode = std::getenv(name);
    return mode && std::string_view(mode) == "1";
}

bool packed_shared_constants_enabled() {
    return env_enabled("INFINILM_MTP_GRAPH_PACKED_SHARED_EXPERT_CONSTANTS");
}

bool k3_draft_seed_enabled() {
    const auto *mode = std::getenv("INFINILM_MTP_GRAPH_K3_DRAFT_SEED");
    return mode && std::string_view(mode) == "1";
}

struct BatchMetadata {
    Tensor position, swa_indices, swa_lengths, swa_slots, page_table;
    Tensor c4_slots, c4_positions, c4_lengths, c4_sparse_lengths, c4_sources, c4_state_slots;
    Tensor c128_slots, c128_positions, c128_indices, c128_lengths, c128_sources, c128_state_slots;
    std::vector<size_t> query_offsets;
    bool packed_decode{};
    bool c4_sequential{};
    size_t c4_query_begin{};
    bool c4_tp_shard{};
    std::optional<C4ScorePlan> c4_score_plan;
    std::optional<RoutedExperts> shared_expert_route{};
    bool packed_decode_collectives{};
    bool packed_decode_mla{};
    unsigned partial_decode_mla_rows{};
    bool packed_shared_constants{};
    bool short_c4_index{};
    bool packed_collectives{};
    bool packed_mhc_pre{};
    bool batched_decode_mhc_pre{};
    bool c128_no_store{};
    bool c128_state_only{};
    bool share_flashmla_schedule{};
    // Derived from the current execution's stable length buffers by Graph nodes.
    // Keep these outputs out of tensors(), which enumerates only incoming copies.
    std::array<Tensor, 3> flashmla_schedules{};
    bool reuse_c4_flashmla_schedule{};
    std::vector<Tensor> c4_flashmla_schedules;
    bool direct_decode_wo_a{};
    bool packed_compressor_projection{};
    bool packed_moe_gate{};
    bool uniform_decode_positions{};

    auto tensors() const {
        return std::array<const Tensor*,17>{&position,&swa_indices,&swa_lengths,&swa_slots,&page_table,
            &c4_slots,&c4_positions,&c4_lengths,&c4_sparse_lengths,&c4_sources,&c4_state_slots,
            &c128_slots,&c128_positions,&c128_indices,&c128_lengths,&c128_sources,&c128_state_slots};
    }
};

constexpr size_t flashmla_schedule_index(size_t ratio) {
    return ratio == 0 ? 0 : ratio == 4 ? 1
                                       : 2;
}

bool short_verify_layout(const std::vector<size_t> &offsets, size_t total_rows) {
    if (offsets.size() < 2 || offsets.size() > 4 ||
        offsets.front() != 0 || offsets.back() != total_rows) {
        return false;
    }
    for (size_t i = 1; i < offsets.size(); ++i) {
        if (offsets[i] <= offsets[i - 1]) {
            return false;
        }
        const auto rows = offsets[i] - offsets[i - 1];
        if (rows < 2 || rows > 4) {
            return false;
        }
    }
    return true;
}

BatchMetadata make_batch(Runtime &rt, const InfinilmModel::Input &input,
                         bool classic_mtp_target,
                         PagedCacheWriteSet *writes = nullptr) {
    const runtime::Device cpu{runtime::Device::Type::kCpu};
    auto host = [&](const std::optional<Tensor> &value) {
        if (!value || !*value) throw std::invalid_argument("DeepSeek V4 requires paged input metadata");
        return (*value)->device() == cpu ? *value : (*value)->to(rt, cpu);
    };
    auto device = [&](const Tensor &t) {
        auto out=Tensor::empty(rt,t->shape(),t->dtype());
        out->copy_from(rt,t);
        return out;
    };
    const auto &prepared = input.deepseek_v4_host_batch;
    const auto positions = prepared ? prepared->positions() : host(input.position_ids);
    const auto offsets = prepared ? prepared->offsets() : host(input.input_offsets);
    const auto *plans = prepared && prepared->plans() ? &*prepared->plans() : nullptr;
    const auto metadata = plans ? plans->attention : engine::build_deepseek_v4_attention_metadata(
        prepared ? prepared->tables() : host(input.block_tables),
        prepared ? prepared->slots() : host(input.slot_mapping),
        positions, offsets, std::nullopt, kV4PageSize, classic_mtp_target);
    const auto c4 = plans ? plans->c4 : engine::build_deepseek_v4_compression_plan(
        metadata, positions, offsets, 4);
    const auto c128 = plans ? plans->c128 : engine::build_deepseek_v4_compression_plan(
        metadata, positions, offsets, 128);
    if (writes) {
        *writes = {metadata.raw_out_loc, metadata.c4_out_loc, metadata.c128_out_loc,
                   c4.state_slots, c128.state_slots};
    }
    const auto *c4_lengths = reinterpret_cast<const int32_t*>(metadata.c4_topk_lengths_raw->data());
    const auto rows = positions->numel();
    const auto *c4_end = c4_lengths + rows;
    const bool hygon = rt.device().type() == runtime::Device::Type::kHygon;
    const auto *first_scored = hygon ? std::find_if(c4_lengths, c4_end, [](int32_t length) {
        return length > static_cast<int32_t>(C4SequentialIndex::kMaxCandidates);
    }) : c4_lengths;
    const bool c4_sequential = hygon && first_scored == c4_end;
    // A single packed request can omit a contiguous prefix. Keep the GEMM
    // layout for multi-request and small speculative verification batches.
    const size_t c4_query_begin = !c4_sequential && offsets->numel() == 2 && rows >= 16
        ? static_cast<size_t>(first_scored - c4_lengths) : 0;
    const auto *row_offsets = reinterpret_cast<const int32_t*>(offsets->data());
    std::vector<size_t> query_offsets(row_offsets, row_offsets + offsets->numel());
    const auto all_past_prefill = [rows](const auto *values) {
        return std::all_of(values, values + rows, [](auto position) { return position > 0; });
    };
    // The body specializes request-local projections by their query spans.
    // Only a full batch with one new token per request uses the decode plan.
    const bool packed_decode = (rows == 1 || hygon) && query_offsets.size() == rows + 1
        && std::adjacent_find(query_offsets.begin(), query_offsets.end(),
                              [](size_t left, size_t right) { return right != left + 1; })
               == query_offsets.end()
        && (positions->dtype() == DataType::kInt64
            ? all_past_prefill(reinterpret_cast<const int64_t*>(positions->data()))
            : all_past_prefill(reinterpret_cast<const int32_t*>(positions->data())));
    // All ranks use the same paged cache layout in fresh prefill. For
    // multiple requests each span has its own short prefix.
    const bool multi_request = query_offsets.size() > 2;
    const bool shard_candidate = hygon && !c4_sequential &&
        (multi_request ? rows >= 4096 && rows <= 32768
                       : c4_query_begin && rows >= 3072 && rows <= 8192 && rows - c4_query_begin >= 512);
    bool c4_tp_shard = false;
    if (shard_candidate) {
        const auto *setting = std::getenv(multi_request ? "INFINILM_DSV4_C4_BATCH_TP_SHARD"
                                                       : "INFINILM_DSV4_C4_TP_SHARD");
        const auto fresh_spans = [&](const auto *positions_data) {
            bool fresh = true;
            for_each_request(query_offsets, [&](const RequestRows &span) {
                fresh = fresh && span.count >= 16 && positions_data[span.first] == 0
                    && positions_data[span.first + span.count - 1] == static_cast<int64_t>(span.count - 1);
            });
            return fresh;
        };
        c4_tp_shard = (!setting || std::string_view(setting) != "0")
            && (positions->dtype() == DataType::kInt64
                ? fresh_spans(reinterpret_cast<const int64_t*>(positions->data()))
                : fresh_spans(reinterpret_cast<const int32_t*>(positions->data())));
    }
    std::optional<C4ScorePlan> c4_score_plan;
    if (c4_tp_shard && multi_request) {
        const auto *setting = std::getenv("INFINILM_DSV4_C4_BALANCED_SHARD");
        if (!setting || std::string_view(setting) != "0") {
            c4_score_plan = make_c4_score_plan(c4_lengths, rows);
        }
    }
    bool c128_no_store = false;
    if (classic_mtp_target && input.mtp_graph_direct_commit && query_offsets.size() == 2 && short_verify_layout(query_offsets, positions->numel()) && rt.device().type() == runtime::Device::Type::kHygon && env_enabled("INFINILM_MTP_GRAPH_K3_C128_STATE_ONLY") && env_enabled("INFINIOPS_V4_C128_POOL_SHORT")) {
        const auto *slots = reinterpret_cast<const int32_t *>(metadata.c128_out_loc->data());
        c128_no_store = std::all_of(slots, slots + metadata.c128_out_loc->numel(),
                                    [](int32_t slot) { return slot == -1; });
    }
    BatchMetadata batch{input.position_ids.value()->view({positions->numel()}),
            device(metadata.swa_indices), device(metadata.swa_topk_lengths), device(metadata.raw_out_loc),
            device(metadata.page_table), device(metadata.c4_out_loc), device(metadata.c4_positions),
            device(metadata.c4_topk_lengths_raw), device(metadata.c4_sparse_topk_lengths),
            device(c4.sources), device(c4.state_slots), device(metadata.c128_out_loc),
            device(metadata.c128_positions), device(metadata.c128_page_indices),
            device(metadata.c128_topk_lengths_clamp1), device(c128.sources), device(c128.state_slots), std::move(query_offsets), packed_decode,
            c4_sequential, c4_query_begin, c4_tp_shard, std::move(c4_score_plan)};
    if (hygon && packed_decode && !classic_mtp_target && rows > 1 && rows <= 16) {
        const auto *mode = std::getenv("INFINILM_DSV4_COMPRESS_PROJECTION");
        batch.packed_compressor_projection = mode && std::string_view(mode) == "packed";
    }
    if (hygon && packed_decode && rows >= 2 && rows <= 8) {
        const auto uniform_positions = [rows](const auto *values) {
            return std::all_of(values + 1, values + rows, [first = values[0]](auto position) { return position == first; });
        };
        batch.uniform_decode_positions = positions->dtype() == DataType::kInt64
            ? uniform_positions(reinterpret_cast<const int64_t*>(positions->data()))
            : uniform_positions(reinterpret_cast<const int32_t*>(positions->data()));
        const auto *setting = std::getenv("INFINILM_DSV4_PACKED_DECODE_MLA");
        if (!setting || std::string_view(setting) == "1") {
            const auto tested_context = [rows](const auto *values) {
                return std::all_of(values, values + rows, [](auto position) { return position < 8192; });
            };
            batch.packed_decode_mla = positions->dtype() == DataType::kInt64
                ? tested_context(reinterpret_cast<const int64_t*>(positions->data()))
                : tested_context(reinterpret_cast<const int32_t*>(positions->data()));
            if (!batch.packed_decode_mla && env_enabled("INFINILM_DSV4_PARTIAL_DECODE_MLA")) {
                const auto eligible = [rows](const auto *values) {
                    unsigned mask = 0;
                    for (size_t row = 0; row < rows; ++row) {
                        if (values[row] < 8192) {
                            mask |= 1u << row;
                        }
                    }
                    // Only contiguous runs of at least two rows can share an MLA call.
                    return mask & ((mask << 1) | (mask >> 1));
                };
                batch.partial_decode_mla_rows = positions->dtype() == DataType::kInt64
                    ? eligible(reinterpret_cast<const int64_t*>(positions->data()))
                    : eligible(reinterpret_cast<const int32_t*>(positions->data()));
            }
        }
    }
    batch.c128_no_store = c128_no_store;
    return batch;
}

class Attention final : public weights::ModelWeights {
public:
    Attention(Runtime &rt, const V4Config &c, const Rank &rank, size_t ratio)
        : ratio_(ratio), heads_(c.heads / rank.tp_size), index_heads_(c.index_heads),
          tp_rank_(rank.tp_rank), tp_size_(rank.tp_size), eps_(c.rms_eps) {
        comm_ = rank.comm;
        const auto bf = DataType::kBFloat16;
        const auto rank_id = size_t(rank.tp_rank), ranks = size_t(rank.tp_size);
        sink_ = bind_weight("attn_sink", Weight(Tensor::empty_parameter(rt,{heads_},DataType::kFloat32),0,rank_id,ranks));
        wq_a_ = bind_child("wq_a",std::make_shared<Linear>(rt,c.hidden,c.q_rank,c.quant,false,bf));
        wkv_ = bind_child("wkv",std::make_shared<Linear>(rt,c.hidden,kV4HeadDim,c.quant,false,bf));
        wq_b_ = bind_child("wq_b",std::make_shared<ColumnLinear>(rt,c.q_rank,c.heads*kV4HeadDim,c.quant,false,bf,rank_id,ranks));
        wo_a_ = bind_child("wo_a",std::make_shared<ColumnLinear>(rt,heads_*kV4HeadDim,c.output_groups*c.o_rank,false,bf,rank_id,ranks));
        wo_b_ = bind_child("wo_b",std::make_shared<RowLinear>(rt,c.output_groups*c.o_rank,c.hidden,c.quant,false,bf,rank_id,ranks,rank.comm));
        q_norm_ = bind_child("q_norm",std::make_shared<RMSNorm>(rt,c.q_rank,c.rms_eps,bf));
        kv_norm_ = bind_child("kv_norm",std::make_shared<RMSNorm>(rt,kV4HeadDim,c.rms_eps,bf));
        q_ones_ = Tensor::empty(rt,{kV4HeadDim},DataType::kFloat32);
        std::vector<float> ones(kV4HeadDim,1.0f);
        rt.memcpyH2D(q_ones_->data(),ones.data(),q_ones_->nbytes(),false);
        if (ratio_) {
            compressor_ = bind_child("compressor",std::make_shared<Compressor>(rt,c.hidden,kV4HeadDim,ratio_,c.rms_eps));
        }
        if (ratio_ == 4) {
            index_compressor_ = bind_child("indexer.compressor",std::make_shared<Compressor>(rt,c.hidden,128,4,c.rms_eps));
            index_query_ = bind_child("indexer.wq_b",std::make_shared<Linear>(rt,c.q_rank,index_heads_*128,c.quant,false,bf));
            index_weights_ = bind_child("indexer.weights_proj",std::make_shared<Linear>(rt,c.hidden,index_heads_,false,bf));
        }
    }

    void process_weights_after_loading(Runtime &rt, bool) override {
        kv_norm_fp32_ = cast(rt,kv_norm_->weight(),DataType::kFloat32);
    }

    void write_swa(Runtime &rt, Tensor x, const BatchMetadata &batch,
                   const LayerPagedCache &cache, const Tensor &frequencies) const {
        auto key = wkv_->forward(rt, x);
        store_swa_rows(rt,key,kv_norm_fp32_,batch.position,frequencies,
                       batch.swa_slots,static_cast<float>(eps_),cache.swa);
    }

    Tensor forward(Runtime &rt, Tensor x, const BatchMetadata &batch, const LayerPagedCache &cache,
                   const Tensor &frequencies) const {
        const auto rows = x->size(0);
        auto q_lora = q_norm_->forward(rt,wq_a_->forward(rt,x));
        auto q_raw = wq_b_->forward(rt,q_lora)->view({rows,heads_,kV4HeadDim});
        const auto eps = static_cast<float>(eps_);
        auto q = normalize_query_and_rotate(rt, q_raw, q_ones_, eps, batch.position, frequencies);
        write_swa(rt,x,batch,cache,frequencies);
        Tensor indices, lengths;
        if (ratio_ == 4) {
            compressor_->store_paged(rt,x,batch.c4_sources,batch.c4_state_slots,
                                     cache.compressor_state,batch.c4_positions,frequencies,
                                     batch.query_offsets,batch.c4_slots,cache.compressed,kV4PageSize/4,
                                     batch.packed_compressor_projection);
            auto key = index_compressor_->forward(rt,x,batch.c4_sources,batch.c4_state_slots,
                                                 cache.indexer_state,batch.c4_positions,frequencies,
                                                 batch.query_offsets,batch.packed_compressor_projection);
            key = cast(rt,key,DataType::kBFloat16);
            if (batch.short_c4_index) {
                // This table is too short for scoring. The key still updates the cache.
                auto query = key->view({rows, 1, kC4IndexWidth});
                auto weights = key->view({key->numel()})->narrow({{0, 0, rows}})->view({rows, 1});
                indices = Tensor::empty(rt,{rows,kC4IndexTopK},DataType::kInt32);
                submit<infini::ops::DeepseekV4C4Index>(rt,key,batch.c4_slots,query,weights,batch.page_table,
                                                      batch.c4_lengths,cache.indexer,indices);
            } else if (batch.c4_sequential) {
                indices = Tensor::empty(rt, {rows, C4SequentialIndex::kMaxCandidates}, DataType::kInt32);
                submit<C4SequentialIndex>(rt, key, batch.c4_slots, batch.page_table,
                                          batch.c4_lengths, cache.indexer, indices);
            } else if (batch.c4_tp_shard && tp_size_ == 8 && index_heads_ == 64
                       && !rt.isGraphRecording()) {
                indices = batch.c4_score_plan
                    ? index_balanced(rt, x, q_lora, key, batch, cache.indexer, frequencies)
                    : index_sharded(rt, x, q_lora, key, batch, cache.indexer, frequencies);
            } else {
                indices = Tensor::empty(rt, {rows, C4SequentialIndex::kMaxCandidates}, DataType::kInt32);
                const auto begin = batch.c4_query_begin;
                const auto scored_rows = rows - begin;
                const auto suffix = [&](const Tensor &value) {
                    return begin ? value->narrow({{0, begin, scored_rows}}) : value;
                };
                auto query_input = suffix(q_lora);
                auto query = index_query_->forward(rt, query_input)->view({scored_rows,index_heads_,128});
                apply_rotary_tail(rt, query, suffix(batch.position), frequencies);
                // Keep the original BF16 GEMM geometry. Changing its number
                // of rows can alter rounding and move nearly tied candidates.
                auto weights = project_requests(rt, batch.query_offsets, [&](const RequestRows &part) {
                    auto input = part.slice(x);
                    return index_weights_->forward(rt, input);
                });
                // A positive scalar common to all candidates does not change top-k.
                submit<infini::ops::DeepseekV4C4Index>(rt, key, batch.c4_slots, query, suffix(weights),
                                                      batch.page_table, batch.c4_lengths,
                                                      cache.indexer, indices);
            }
            lengths = batch.c4_sparse_lengths;
        } else if (ratio_ == 128) {
            if (batch.c128_state_only) {
                compressor_->update_state(rt, x, batch.c128_state_slots, cache.compressor_state,
                                          batch.query_offsets,batch.packed_compressor_projection);
            } else {
                compressor_->store_paged(rt,x,batch.c128_sources,batch.c128_state_slots,
                                         cache.compressor_state,batch.c128_positions,frequencies,
                                         batch.query_offsets,batch.c128_slots,cache.compressed,kV4PageSize/128,
                                         batch.packed_compressor_projection);
            }
            indices = batch.c128_indices;
            lengths = batch.c128_lengths;
        }
        const auto attend = [&](const RequestRows &part) {
            const auto &schedule = ratio_ == 4 && batch.reuse_c4_flashmla_schedule
                ? batch.c4_flashmla_schedules.at(part.first)
                : batch.flashmla_schedules[flashmla_schedule_index(ratio_)];
            return sparse_mla(rt, part.slice(q), cache, part.slice(batch.swa_indices),
                              part.slice(batch.swa_lengths), sink_, part.slice(indices),
                              part.slice(lengths), 1.0 / std::sqrt(double(kV4HeadDim)), schedule);
        };
        Tensor packed_attention;
        std::vector<std::pair<size_t, Tensor>> prefill_attention_parts;
        // C4 can repartition the FP32 split-KV reduction when the batch grows.
        // Packed decode MLA is limited to the tested SWA and C128 geometries.
        const bool packable_decode_mla = q->dtype() == DataType::kBFloat16 && heads_ == 8 &&
            tp_size_ == 8 && (ratio_ == 0 || ratio_ == 128);
        if (batch.packed_decode_mla && packable_decode_mla) {
            packed_attention = attend({0, rows});
        } else if (rt.device().type() == runtime::Device::Type::kHygon && !rt.isGraphRecording()
            && q->dtype() == DataType::kBFloat16 && heads_ == 8 && tp_size_ == 8
            && batch.query_offsets.size() > 2) {
            constexpr size_t min_request_rows = 4096;
            constexpr size_t max_group_rows = 32768;
            const auto *setting = std::getenv("INFINILM_DSV4_PACKED_PREFILL_MLA");
            if (!setting || std::string_view(setting) != "0") {
                bool long_requests = true;
                for_each_request(batch.query_offsets, [&](const RequestRows &part) {
                    long_requests = long_requests && part.count >= min_request_rows;
                });
                if (long_requests && rows <= max_group_rows) {
                    packed_attention = attend({0, rows});
                } else if (env_enabled("INFINILM_DSV4_PACKED_RAGGED_PREFILL_MLA")) {
                    std::vector<RequestRows> group;
                    size_t group_rows = 0;
                    const auto submit_group = [&] {
                        if (group.size() >= 2) {
                            const auto &first = group.front();
                            const auto &last = group.back();
                            Tensor attended;
                            if (last.first + last.count - first.first == group_rows) {
                                attended = attend({first.first, group_rows});
                            } else {
                                const auto pack = [&](const Tensor &input) -> Tensor {
                                    if (!input) {
                                        return {};
                                    }
                                    auto shape = input->shape();
                                    shape[0] = group_rows;
                                    auto packed = Tensor::empty(rt, shape, input->dtype());
                                    size_t offset = 0;
                                    for (const auto &part : group) {
                                        const RequestRows target{offset, part.count};
                                        target.slice(packed)->copy_from(rt, part.slice(input));
                                        offset += part.count;
                                    }
                                    return packed;
                                };
                                attended = sparse_mla(rt, pack(q), cache, pack(batch.swa_indices),
                                                      pack(batch.swa_lengths), sink_, pack(indices),
                                                      pack(lengths), 1.0 / std::sqrt(double(kV4HeadDim)), {});
                            }
                            // Keep request order for RoPE and the shape-sensitive wo_a.
                            size_t offset = 0;
                            for (const auto &part : group) {
                                const RequestRows source{offset, part.count};
                                prefill_attention_parts.emplace_back(part.first, source.slice(attended));
                                offset += part.count;
                            }
                        }
                        group.clear();
                        group_rows = 0;
                    };
                    for_each_request(batch.query_offsets, [&](const RequestRows &part) {
                        if (part.count < min_request_rows || part.count > max_group_rows) {
                            return;
                        }
                        if (group_rows + part.count > max_group_rows) {
                            submit_group();
                        }
                        group.push_back(part);
                        group_rows += part.count;
                    });
                    submit_group();
                }
            }
        }
        std::array<Tensor, 8> partial_attention{};
        if (batch.partial_decode_mla_rows && packable_decode_mla) {
            const auto mask = batch.partial_decode_mla_rows;
            for (size_t first = 0; first < rows;) {
                if (!(mask & (1u << first))) {
                    ++first;
                    continue;
                }
                size_t end = first + 1;
                while (end < rows && (mask & (1u << end))) {
                    ++end;
                }
                auto grouped = attend({first, end - first});
                for (size_t row = first; row < end; ++row) {
                    partial_attention[row] = grouped->narrow({{0, row - first, 1}});
                }
                first = end;
            }
        }
        const auto attend_request = [&](const RequestRows &part) {
            if (packed_attention) {
                return part.slice(packed_attention);
            }
            if (batch.partial_decode_mla_rows && partial_attention[part.first]) {
                return partial_attention[part.first];
            }
            return attend(part);
        };
        Tensor local;
        if (batch.direct_decode_wo_a && tp_size_ == 8 && heads_ == 8 &&
            wo_a_->weight()->size(0) == 1024) {
            const auto weight = wo_a_->weight()->contiguous(rt);
            local = Tensor::empty(rt, {rows, weight->size(0)}, weight->dtype());
            // Keep request-local rotary execution in mixed-position batches.
            const bool packed_rope = packed_attention && batch.uniform_decode_positions;
            if (packed_rope) {
                apply_rotary_tail(rt, packed_attention, batch.position, frequencies, true);
            }
            // Keep the individual BLAS calls. Changing the batch count can
            // change the accumulation even while each logical GEMM remains M=1.
            for_each_request(batch.query_offsets, [&](const RequestRows &part) {
                auto attended = attend_request(part);
                if (!packed_rope) {
                    apply_rotary_tail(rt, attended, part.slice(batch.position), frequencies, true);
                }
                auto flattened = attended->view({part.count, heads_ * kV4HeadDim});
                layers::linear::dense_projection(rt, flattened, weight, std::nullopt, wo_a_->alpha(),
                                                 std::nullopt, false, part.slice(local));
            });
        } else {
            auto prefill = prefill_attention_parts.begin();
            local = project_requests(rt, batch.query_offsets, [&](const RequestRows &part) {
                auto attended = prefill != prefill_attention_parts.end() && prefill->first == part.first
                    ? (prefill++)->second : attend_request(part);
                apply_rotary_tail(rt, attended, part.slice(batch.position), frequencies, true);
                auto flattened = attended->view({part.count, heads_ * kV4HeadDim});
                return wo_a_->forward(rt, flattened);
            });
        }
        auto contribution = static_cast<const nn::BaseLinear&>(*wo_b_).forward(rt,local);
        return reduce_tp(rt, contribution, comm_, batch.query_offsets,
                         batch.packed_decode_collectives || batch.packed_collectives);
    }

private:
    Tensor index_balanced(Runtime &rt, const Tensor &x, const Tensor &q_lora, const Tensor &key,
                          const BatchMetadata &batch, const Tensor &cache,
                          const Tensor &frequencies) const {
        constexpr auto top = C4SequentialIndex::kMaxCandidates;
        const auto &plan = *batch.c4_score_plan;
        const auto &spans = plan.scored[tp_rank_];
        const auto count = plan.row_counts[tp_rank_];
        auto output = Tensor::empty(rt, {x->size(0), top}, DataType::kInt32);
        const auto empty_key = key->narrow({{0, 0, 0}});
        const auto empty_slots = batch.c4_slots->narrow({{0, 0, 0}});

        for (size_t i = 0; i < plan.sequential.size(); ++i) {
            const auto &span = plan.sequential[i];
            submit<C4SequentialIndex>(rt, i == 0 ? key : empty_key,
                i == 0 ? batch.c4_slots : empty_slots, span.slice(batch.page_table),
                span.slice(batch.c4_lengths), cache, span.slice(output));
        }

        const auto pack = [&](const Tensor &tensor) {
            auto shape = tensor->shape();
            shape[0] = count;
            auto packed = Tensor::empty(rt, shape, tensor->dtype());
            size_t first = 0;
            for (const auto &span : spans) {
                packed->narrow({{0, first, span.count}})->copy_from(rt, span.slice(tensor));
                first += span.count;
            }
            return packed;
        };
        const bool project_local = env_enabled("INFINILM_DSV4_C4_PROJECT_LOCAL");
        auto query_input = project_local ? pack(q_lora) : q_lora;
        auto projected = index_query_->forward(rt, query_input);
        auto query = (project_local ? projected : pack(projected))->view({count, index_heads_, 128});
        auto positions = pack(batch.position);
        apply_rotary_tail(rt, query, positions, frequencies);
        auto weights = project_requests(rt, batch.query_offsets, [&](const RequestRows &part) {
            auto input = part.slice(x);
            return index_weights_->forward(rt, input);
        });
        auto local = Tensor::empty(rt, {plan.gathered_rows, top}, DataType::kInt32);
        submit<infini::ops::DeepseekV4C4Index>(rt, empty_key, empty_slots, query, pack(weights),
            pack(batch.page_table), pack(batch.c4_lengths), cache, local->narrow({{0, 0, count}}));
        if (count < plan.gathered_rows) {
            auto padding = local->narrow({{0, count, plan.gathered_rows - count}});
            rt.setDeviceMemoryAsync(padding->data(), 0, padding->nbytes(), rt.stream());
        }

        auto gathered = Tensor::empty(rt, {C4ScorePlan::ranks * plan.gathered_rows, top}, DataType::kInt32);
        engine::distributed::allgather_(rt, gathered, local, comm_);
        for (size_t rank = 0; rank < C4ScorePlan::ranks; ++rank) {
            size_t first = rank * plan.gathered_rows;
            for (const auto &span : plan.scored[rank]) {
                span.slice(output)->copy_from(rt, gathered->narrow({{0, first, span.count}}));
                first += span.count;
            }
        }
        return output;
    }

    Tensor index_sharded(Runtime &rt, const Tensor &x, const Tensor &q_lora, const Tensor &key,
                         const BatchMetadata &batch, const Tensor &cache,
                         const Tensor &frequencies) const {
        constexpr auto top = C4SequentialIndex::kMaxCandidates;
        const auto rows = x->size(0);
        const auto prefix_rows = batch.c4_query_begin;
        const auto scored_rows = rows - prefix_rows;
        const auto shard_rows = (scored_rows + tp_size_ - 1) / tp_size_;
        const auto shard_offset = tp_rank_ * shard_rows;
        const auto first = prefix_rows + shard_offset;
        const auto count = std::min(shard_rows, rows - first);
        const auto take = [](const Tensor &tensor, size_t begin, size_t count) {
            return tensor->narrow({{0, begin, count}});
        };

        const auto gathered_rows = tp_size_ * shard_rows;
        auto gathered = Tensor::empty(rt, {prefix_rows + gathered_rows, top}, DataType::kInt32);
        // The sequential provider writes all keys and can emit the existing
        // common prefix. For packed requests the gathered rows cover it all.
        const auto write_rows = std::max<size_t>(prefix_rows, 1);
        submit<C4SequentialIndex>(rt, key, batch.c4_slots, take(batch.page_table, 0, write_rows),
                                  take(batch.c4_lengths, 0, write_rows), cache,
                                  take(gathered, 0, write_rows));

        const bool multi_request = batch.query_offsets.size() > 2;
        // Hygon can project only this rank's W8A8 query rows. The BF16
        // index weight projection remains request-local.
        const bool project_full = multi_request && !env_enabled("INFINILM_DSV4_C4_PROJECT_LOCAL");
        auto query_input = project_full ? q_lora : take(q_lora, first, count);
        auto projected = index_query_->forward(rt, query_input);
        auto query = (project_full ? take(projected, first, count) : projected)
                         ->view({count, index_heads_, 128});
        apply_rotary_tail(rt, query, take(batch.position, first, count), frequencies);
        auto weights = project_requests(rt, batch.query_offsets, [&](const RequestRows &part) {
            auto input = part.slice(x);
            return index_weights_->forward(rt, input);
        });
        auto local = Tensor::empty(rt, {shard_rows, top}, DataType::kInt32);
        submit<infini::ops::DeepseekV4C4Index>(
            rt, take(key, 0, 0), take(batch.c4_slots, 0, 0), query, take(weights, first, count),
            take(batch.page_table, first, count), take(batch.c4_lengths, first, count), cache,
            take(local, 0, count));
        if (count < shard_rows) {
            // The collective needs equal counts. Padding never reaches the attention op.
            auto padding = take(local, count, shard_rows - count);
            rt.setDeviceMemoryAsync(padding->data(), 0, padding->nbytes(), rt.stream());
        }
        engine::distributed::allgather_(rt, take(gathered, prefix_rows, gathered_rows),
                                       local, comm_);
        return take(gathered, 0, rows);
    }

    infinicclComm_t comm_;
    size_t ratio_, heads_, index_heads_;
    size_t tp_rank_, tp_size_;
    double eps_;
    Tensor sink_, q_ones_, kv_norm_fp32_;
    std::shared_ptr<Linear> wq_a_, wkv_, index_query_, index_weights_;
    std::shared_ptr<ColumnLinear> wq_b_, wo_a_;
    std::shared_ptr<RowLinear> wo_b_;
    std::shared_ptr<RMSNorm> q_norm_, kv_norm_;
    std::shared_ptr<Compressor> compressor_, index_compressor_;
};

RoutedExperts make_shared_expert_route(Runtime &rt, size_t rows) {
    auto weights = Tensor::empty(rt, {rows, 1}, DataType::kFloat32);
    auto indices = Tensor::empty(rt, {rows, 1}, DataType::kInt64);
    submit<infini::ops::Fill>(rt, weights, 1.0, weights);
    submit<infini::ops::Fill>(rt, indices, 0.0, indices);
    return {std::move(weights), std::move(indices)};
}

class FeedForward final : public weights::ModelWeights {
public:
    FeedForward(Runtime &rt, const V4Config &c, const Rank &rank, size_t layer)
        : topk_(c.topk), scale_(c.route_scale), limit_(c.swiglu_limit), comm_(rank.comm) {
        const auto bf = DataType::kBFloat16;
        gate_ = bind_child("gate",std::make_shared<Linear>(rt,c.hidden,c.experts,false,bf,DataType::kFloat32));
        if (layer < c.hash_layers) {
            hash_ = bind_weight("gate.tid2eid",Weight(rt,{c.vocab,c.topk},DataType::kInt64));
        } else {
            bias_ = bind_weight("gate.bias",Weight(rt,{c.experts},DataType::kFloat32));
        }
        experts_ = bind_child("experts",std::make_shared<W8A8ExpertWeights>(
            rt,c.experts,c.hidden,c.intermediate,rank.tp_rank,rank.tp_size));
        shared_one_ = Tensor::empty(rt,{1,1},DataType::kFloat32);
        shared_zero_ = Tensor::empty(rt,{1,1},DataType::kInt64);
        const float one=1.f;
        const int64_t zero=0;
        rt.memcpyH2D(shared_one_->data(),&one,sizeof(one),false);
        rt.memcpyH2D(shared_zero_->data(),&zero,sizeof(zero),false);
        if ((c.classic_mtp_target && !c.is_mtp && packed_shared_constants_enabled())
            || (c.is_mtp && k3_draft_seed_enabled())) {
            const Shape shape{kPackedSharedExpertRows, 1};
            std::array<float, kPackedSharedExpertRows> ones;
            ones.fill(1.f);
            const std::array<int64_t, kPackedSharedExpertRows> zeros{};
            packed_shared_ones_ = Tensor::empty(rt, shape, DataType::kFloat32);
            packed_shared_zeros_ = Tensor::empty(rt, shape, DataType::kInt64);
            rt.memcpyH2D(packed_shared_ones_->data(), ones.data(), sizeof(ones), false);
            rt.memcpyH2D(packed_shared_zeros_->data(), zeros.data(), sizeof(zeros), false);
        }
        shared_ = bind_child("shared_experts",std::make_shared<W8A8ExpertWeights>(
            rt,1,c.hidden,c.intermediate,rank.tp_rank,rank.tp_size,true));
    }

    Tensor forward(Runtime &rt, Tensor x, const Tensor &tokens, const BatchMetadata &batch) const {
        auto logits = batch.packed_moe_gate
            ? gate_->forward(rt, x)
            : project_requests(rt, batch.query_offsets, [&](const RequestRows &part) {
                  auto input = part.slice(x);
                  return gate_->forward(rt, input);
              });
        using Opt=std::optional<Tensor>;
        const Shape routes{x->size(0),topk_};
        RoutedExperts route{Tensor::empty(rt,routes,DataType::kFloat32),
                            Tensor::empty(rt,routes,DataType::kInt64)};
        submit<infini::ops::DeepseekV4Route>(rt,logits,bias_ ? Opt{bias_} : Opt{},
            hash_ ? Opt{tokens} : Opt{},hash_ ? Opt{hash_} : Opt{},1.0,route.weights,route.indices);
        auto routed = run_routed_experts(rt,x,route,*experts_,limit_,scale_);
        auto shared_route = batch.shared_expert_route ? *batch.shared_expert_route
            : batch.packed_shared_constants || x->size(0) == 1 ? RoutedExperts{shared_one_, shared_zero_}
                              : make_shared_expert_route(rt, x->size(0));
        if (batch.packed_shared_constants) {
            const auto rows = x->size(0);
            // The shared expert always consumes weight 1 and expert ID 0.
            auto ones = packed_shared_ones_->narrow({{0, 0, rows}});
            auto zero = packed_shared_zeros_->narrow({{0, 0, rows}});
            shared_route = {std::move(ones), std::move(zero)};
        }
        auto shared = run_routed_experts(rt, x, shared_route, *shared_, limit_, std::nullopt);
        auto sum = add(rt,routed,shared);
        return reduce_tp(rt, sum, comm_, batch.query_offsets,
                         batch.packed_decode_collectives || batch.packed_collectives);
    }

private:
    size_t topk_;
    double scale_, limit_;
    infinicclComm_t comm_;
    Tensor hash_, bias_, shared_one_, shared_zero_;
    Tensor packed_shared_ones_, packed_shared_zeros_;
    std::shared_ptr<Linear> gate_;
    std::shared_ptr<W8A8ExpertWeights> experts_, shared_;
};

HyperConnectionMix run_mhc(Runtime &rt, const Tensor &x, const MixWeights &w,
                           double norm_eps, double mix_eps, size_t iterations,
                           const BatchMetadata &batch, size_t *packed_mhc_calls) {
    const auto rows = x->size(0), streams = x->size(1);
    HyperConnectionMix out{Tensor::empty(rt, {rows, x->size(2)}, x->dtype()),
                           Tensor::empty(rt, {rows, streams}, DataType::kFloat32),
                           Tensor::empty(rt, {rows, streams, streams}, DataType::kFloat32)};
    const auto submit_mix = [&](const Tensor &input, const Tensor &activation,
                                const Tensor &post, const Tensor &combination) {
        submit<infini::ops::DeepseekV4MhcPre>(
            rt, input, w.projection, w.scale, w.base, norm_eps, mix_eps,
            static_cast<int64_t>(iterations), activation, post, combination);
    };
    if (batch.batched_decode_mhc_pre && x->dtype() == DataType::kBFloat16 && x->is_contiguous() &&
        rows >= 2 && rows <= 8 && streams == 4 && x->size(2) == 4096 &&
        reinterpret_cast<uintptr_t>(x->data()) % 8 == 0 &&
        reinterpret_cast<uintptr_t>(w.projection->data()) % 16 == 0) {
        if (iterations == 0) {
            // The head already applies its single-row reduction independently.
            submit_mix(x, out.activation, out.post, out.combination);
        } else {
            using Op = infini::ops::DeepseekV4MhcPre;
            infini::ops::DeepseekV4MhcPreSingleTokenBatchConfig config;
            config.set_implementation_index(Op::DefaultConfig(rt.device().type()).implementation_index());
            execution::submit<Op>(rt, execution::RecordingPolicy::DeviceOnly, config,
                                  x, w.projection, w.scale, w.base, norm_eps, mix_eps,
                                  static_cast<int64_t>(iterations), out.activation, out.post, out.combination);
        }
        return out;
    }
    // Flash BF16 q2–q4 and packed M6–M12 use the same per-row reduction.
    // A contiguous row preserves the alignment of every owner slice.
    if (batch.packed_mhc_pre && rt.device().type() == runtime::Device::Type::kHygon && x->ndim() == 3 && x->dtype() == DataType::kBFloat16 && x->is_contiguous() && rows >= 6 && rows <= 12 && streams == 4 && x->size(2) == 4096) {
        submit_mix(x, out.activation, out.post, out.combination);
        if (packed_mhc_calls) {
            ++*packed_mhc_calls;
        }
        return out;
    }
    for_each_request(batch.query_offsets, [&](const RequestRows &part) {
        submit_mix(part.slice(x), part.slice(out.activation),
                   part.slice(out.post), part.slice(out.combination));
    });
    return out;
}

Tensor run_mhc_post(Runtime &rt,const Tensor &x,const Tensor &residual,const HyperConnectionMix &mix) {
    auto out=Tensor::empty(rt,residual->shape(),residual->dtype());
    submit<infini::ops::DeepseekV4MhcPost>(rt,x,residual,mix.post,mix.combination,out);
    return out;
}

class Decoder final : public weights::ModelWeights {
public:
    Decoder(Runtime &rt,const V4Config &c,const Rank &rank,size_t layer)
        : norm_eps_(c.rms_eps), mix_eps_(c.hc_eps), iterations_(c.iterations) {
        const size_t mix = (2+c.streams)*c.streams;
        const auto fp = DataType::kFloat32;
        auto bind_mix = [&](const std::string &prefix) {
            return MixWeights{bind_weight(prefix+"_fn",Weight(rt,{mix,c.streams*c.hidden},fp)),
                              bind_weight(prefix+"_scale",Weight(rt,{3},fp)),
                              bind_weight(prefix+"_base",Weight(rt,{mix},fp))};
        };
        attn_mix_ = bind_mix("hc_attn");
        ffn_mix_ = bind_mix("hc_ffn");
        attn_norm_ = bind_child("attn_norm",std::make_shared<RMSNorm>(rt,c.hidden,c.rms_eps,DataType::kBFloat16));
        ffn_norm_ = bind_child("ffn_norm",std::make_shared<RMSNorm>(rt,c.hidden,c.rms_eps,DataType::kBFloat16));
        attn_ = bind_child("attn",std::make_shared<Attention>(rt,c,rank,c.ratios.at(layer)));
        ffn_ = bind_child("ffn",std::make_shared<FeedForward>(rt,c,rank,layer));
    }

    Tensor forward(Runtime &rt, const Tensor &residual, const Tensor &tokens,
                   const BatchMetadata &batch, const LayerPagedCache &cache, const Tensor &frequencies,
                   size_t *packed_mhc_calls, bool cache_write_only) const {
        const auto attention = mix(rt, residual, attn_mix_, batch, packed_mhc_calls);
        const auto normalized = attn_norm_->forward(rt, attention.activation);
        if (cache_write_only) {
            attn_->write_swa(rt, normalized, batch, cache, frequencies);
            return {};
        }
        const auto attended = attn_->forward(rt, normalized, batch, cache, frequencies);
        const auto next = run_mhc_post(rt,attended,residual,attention);
        const auto feed_forward = mix(rt, next, ffn_mix_, batch, packed_mhc_calls);
        const auto experts = ffn_->forward(rt, ffn_norm_->forward(rt, feed_forward.activation), tokens, batch);
        return run_mhc_post(rt,experts,next,feed_forward);
    }

private:
    HyperConnectionMix mix(Runtime &rt, const Tensor &x, const MixWeights &w,
                           const BatchMetadata &batch, size_t *packed_mhc_calls) const {
        return run_mhc(rt, x, w, norm_eps_, mix_eps_, iterations_, batch, packed_mhc_calls);
    }
    double norm_eps_, mix_eps_;
    size_t iterations_;
    MixWeights attn_mix_, ffn_mix_;
    std::shared_ptr<RMSNorm> attn_norm_, ffn_norm_;
    std::shared_ptr<Attention> attn_;
    std::shared_ptr<FeedForward> ffn_;
};

class Backbone final : public weights::ModelWeights {
public:
    Backbone(Runtime &rt,const V4Config &c,const Rank &rank)
        : hidden_(c.hidden), streams_(c.streams), rms_eps_(c.rms_eps), hc_eps_(c.hc_eps), is_mtp_(c.is_mtp) {
        embedding_ = bind_child("embed_tokens",std::make_shared<layers::Embedding>(
            rt,c.vocab,c.hidden,std::nullopt,DataType::kBFloat16));
        norm_ = bind_child("norm",std::make_shared<RMSNorm>(rt,c.hidden,c.rms_eps,DataType::kBFloat16));
        if (is_mtp_) {
            enorm_ = bind_child("enorm",std::make_shared<RMSNorm>(rt,c.hidden,c.rms_eps,DataType::kBFloat16));
            hnorm_ = bind_child("hnorm",std::make_shared<RMSNorm>(rt,c.hidden,c.rms_eps,DataType::kBFloat16));
            e_proj_ = bind_child("e_proj",std::make_shared<Linear>(rt,c.hidden,c.hidden,c.quant,false,DataType::kBFloat16));
            h_proj_ = bind_child("h_proj",std::make_shared<Linear>(rt,c.hidden,c.hidden,c.quant,false,DataType::kBFloat16));
        }
        const auto fp = DataType::kFloat32;
        head_ = {bind_weight("hc_head_fn",Weight(rt,{c.streams,c.streams*c.hidden},fp)),
                 bind_weight("hc_head_scale",Weight(rt,{1},fp)),
                 bind_weight("hc_head_base",Weight(rt,{c.streams},fp))};
        for (size_t i=0;i<c.ratios.size();++i) {
            layers_.push_back(bind_child("layers."+std::to_string(i),std::make_shared<Decoder>(rt,c,rank,i)));
        }
    }

    std::pair<Tensor, Tensor> forward(Runtime &rt, Tensor tokens, const BatchMetadata &batch,
                                      const std::vector<LayerPagedCache> &caches,
                                      const Tensor &swa_rope, const Tensor &compressed_rope,
                                      const Tensor &target_hc, bool return_hc, bool cache_only = false,
                                      size_t *packed_mhc_calls = nullptr, bool eager_cache_write_only = false) const {
        const auto rows = tokens->numel();
        const bool cache_write_only = eager_cache_write_only || (cache_only && batch.query_offsets.size() != 2);
        if (cache_write_only) {
            constexpr std::array<size_t, 2> eager_offsets{0, 1};
            constexpr std::array<size_t, 4> packed_offsets{0, 1, 2, 3};
            const auto &offsets = batch.query_offsets;
            const bool layout_ok = eager_cache_write_only
                                     ? rows == 1 && std::equal(offsets.begin(), offsets.end(), eager_offsets.begin(), eager_offsets.end())
                                     : rows == 3 && std::equal(offsets.begin(), offsets.end(), packed_offsets.begin(), packed_offsets.end());
            if (!cache_only || !is_mtp_ || layers_.size() != 1 || caches.size() != 1 || caches.front().compression_ratio != 0 || !layout_ok) {
                throw std::invalid_argument("Classic NextN output-free repair requires one SWA layer and a supported q1 layout");
            }
        }
        const Shape hc_shape{rows,streams_,hidden_};
        if (is_mtp_ && (!target_hc || target_hc->shape() != hc_shape ||
                        target_hc->dtype() != DataType::kBFloat16 || target_hc->device() != rt.device())) {
            throw std::invalid_argument("Classic NextN requires rank-local BF16 Target HC [tokens,4,hidden]");
        }
        auto embedded = embedding_->forward(rt,tokens)->view({rows,hidden_});
        const auto expand = [&](const Tensor &value) {
            return value->as_strided(hc_shape, {static_cast<runtime::Stride>(hidden_),0,1})->contiguous(rt);
        };
        Tensor residual;
        if (is_mtp_) {
            auto normalized_embedding = enorm_->forward(rt,embedded);
            auto e = e_proj_->forward(rt,normalized_embedding);
            // The caller completes the Target producer stream before dispatch.
            auto hc = Tensor::empty(rt,hc_shape,DataType::kBFloat16);
            hc->copy_from(rt,target_hc);
            auto normalized_hc = hnorm_->forward(rt,hc->view({rows*streams_,hidden_}));
            auto h = h_proj_->forward(rt,normalized_hc);
            residual = add(rt,h->view(hc_shape),expand(e));
        } else {
            residual = expand(embedded);
        }
        for (size_t i=0;i<layers_.size();++i) {
            residual = layers_[i]->forward(rt, residual, tokens, batch, caches[i],
                                           caches[i].compression_ratio ? compressed_rope : swa_rope,
                                           packed_mhc_calls, cache_write_only);
        }
        if (cache_only) {
            return {};
        }
        Tensor hc_output;
        if (return_hc) {
            hc_output = Tensor::empty(rt,hc_shape,DataType::kBFloat16);
            hc_output->copy_from(rt,residual);
        }
        auto normalized = norm_->forward(rt, run_mhc(
                                                 rt, residual, head_, rms_eps_, hc_eps_, 0, batch, packed_mhc_calls)
                                                 .activation);
        return {std::move(normalized),std::move(hc_output)};
    }

private:
    size_t hidden_, streams_;
    double rms_eps_, hc_eps_;
    bool is_mtp_;
    MixWeights head_;
    std::shared_ptr<layers::Embedding> embedding_;
    std::shared_ptr<RMSNorm> norm_, enorm_, hnorm_;
    std::shared_ptr<Linear> e_proj_, h_proj_;
    std::vector<std::shared_ptr<Decoder>> layers_;
};

class DeepseekV4ForCausalLM final : public InfinilmModel {
public:
    DeepseekV4ForCausalLM(Runtime &rt,std::shared_ptr<ModelConfig> config,const Rank &rank,
                         backends::AttentionBackend backend,const layers::moe::EPConfig &ep)
        : cfg_(*config,rank), comm_(rank.comm), sampling_rank_(rank.tp_rank == 0) {
        validate_runtime_construction(config,rt,rank);
        if (backend != backends::AttentionBackend::PAGED_ATTN || ep.backend != layers::moe::EPBackend::Disabled)
            throw std::invalid_argument("DeepSeek V4 requires paged attention and TP expert placement");
        model_config_ = std::move(config);
        const auto *reuse = std::getenv("INFINILM_MTP_REUSE_SNAPSHOT_BUFFER");
        reuse_snapshot_buffer_ = reuse && std::string_view(reuse) == "1";
        const auto *k1_seed = std::getenv("INFINILM_MTP_K1_GRAPH_SEED");
        k1_graph_seed_ = cfg_.is_mtp && k1_seed && std::string_view(k1_seed) == "1";
        packed_shared_constants_prepared_ = cfg_.classic_mtp_target && !cfg_.is_mtp && packed_shared_constants_enabled();
        draft_seed_constants_prepared_ = cfg_.is_mtp && k3_draft_seed_enabled();
        backbone_ = bind_child("model",std::make_shared<Backbone>(rt,cfg_,rank));
        head_ = bind_child("lm_head",std::make_shared<ColumnLinear>(
            rt,cfg_.hidden,cfg_.vocab,false,DataType::kBFloat16,rank.tp_rank,rank.tp_size));
    }

    Output forward(const Input &input, global_state::ForwardContext &ctx) const override {
        auto &rt = ctx.require_runtime();
        if (caches_.empty() || !input.input_ids || !*input.input_ids || rt.isGraphRecording())
            throw std::invalid_argument("DeepSeek V4 requires initialized paged cache and eager input preparation");
        auto ids = (*input.input_ids)->view({(*input.input_ids)->numel()});
        PagedCacheWriteSet writes;
        auto batch = make_batch(rt, input, cfg_.classic_mtp_target,
                                input.rollback_v4_cache ? &writes : nullptr);
        if (!ctx.model_decode_barrier || cfg_.is_mtp) {
            batch.packed_decode_mla = false;
            batch.partial_decode_mla_rows = 0;
        }
        if (input.mtp_eager_packed_direct_commit) {
            if (!cfg_.classic_mtp_target || cfg_.is_mtp || input.target_hidden_states
                || !input.return_draft_hidden_states || !input.sample_all_positions
                || input.rollback_v4_cache || input.mtp_scratch_step || input.mtp_draft_cache_only
                || input.mtp_last_logits || input.mtp_graph_direct_commit || input.mtp_graph_draft_seed
                || ctx.model_decode_barrier || ctx.mtp_graph_barrier || ctx.mtp_snapshot_barrier
                || batch.query_offsets.size() != 4 || !short_verify_layout(batch.query_offsets, ids->numel())) {
                throw std::invalid_argument("Packed eager direct MTP requires complete three-owner Target queries");
            }
            const auto &modes = input.mtp_eager_target_modes;
            batch.packed_collectives = modes.packed_collectives;
            batch.packed_mhc_pre = modes.packed_mhc_pre;
            const auto table_width = batch.page_table->size(1);
            batch.short_c4_index = modes.short_c4_index
                                && rt.device().type() == runtime::Device::Type::kHygon
                                && table_width > 0 && table_width <= kC4IndexMaxUnscoredPages;
            size_t packed_mhc_calls = 0;
            auto output = forward_logits(rt, ids, batch, {}, true, false,
                                         modes.packed_mhc_pre ? &packed_mhc_calls : nullptr);
            const auto &prepared = input.deepseek_v4_host_batch;
            output.mtp_eager_target_receipt = EagerPackedTargetModes{
                batch.packed_collectives,
                prepared && bool(prepared->plans()),
                packed_mhc_calls == cfg_.ratios.size() * 2 + 1,
                batch.short_c4_index
                    && std::find(cfg_.ratios.begin(), cfg_.ratios.end(), 4) != cfg_.ratios.end(),
            };
            return output;
        }
        if (input.mtp_draft_cache_only) {
            if (!cfg_.is_mtp || input.return_draft_hidden_states || input.rollback_v4_cache ||
                ctx.model_decode_barrier || ctx.mtp_graph_barrier) {
                throw std::invalid_argument("Classic NextN cache-only is an eager Draft write");
            }
            backbone_->forward(rt, ids, batch, caches_, swa_rope_, compressed_rope_,
                               input.target_hidden_states.value_or(Tensor{}), false, true,
                               nullptr, input.mtp_eager_c3_draft_repair_cache_only);
            Output output{};
            output.mtp_draft_cache_only_used = true;
            return output;
        }
        if (ctx.model_decode_barrier && batch.packed_decode) {
            const auto *packed = std::getenv("INFINILM_DSV4_PACKED_DECODE");
            batch.packed_decode_collectives = !cfg_.is_mtp && ids->numel() > 1 &&
                rt.device().type() == runtime::Device::Type::kHygon &&
                (!packed || std::string_view(packed) != "0");
            batch.packed_moe_gate = !cfg_.is_mtp && !cfg_.classic_mtp_target &&
                ids->numel() >= 2 && ids->numel() <= 16 &&
                rt.device().type() == runtime::Device::Type::kHygon &&
                env_enabled("INFINILM_DSV4_PACK_MOE_GATE");
            batch.reuse_c4_flashmla_schedule = batch.uniform_decode_positions && !cfg_.is_mtp && ids->numel() >= 2
                && ids->numel() <= 8 && cfg_.heads / kV4TensorParallelSize == 8
                && rt.device().type() == runtime::Device::Type::kHygon
                && env_enabled("INFINILM_DSV4_C4_FLASHMLA_SCHEDULE")
                && std::find(cfg_.ratios.begin(), cfg_.ratios.end(), 4) != cfg_.ratios.end();
            batch.direct_decode_wo_a = batch.packed_decode_collectives && ids->numel() <= 8 &&
                env_enabled("INFINILM_DSV4_DIRECT_WO_A") &&
                (batch.uniform_decode_positions || env_enabled("INFINILM_DSV4_RAGGED_WO_A"));
            batch.batched_decode_mhc_pre = batch.packed_decode_collectives && ids->numel() <= 8 &&
                env_enabled("INFINILM_DSV4_BATCHED_MHC_PRE") &&
                (batch.uniform_decode_positions || env_enabled("INFINILM_DSV4_RAGGED_MHC_PRE"));
            const bool fused_greedy = ctx.model_decode_greedy && ids->numel() == 1;
            return decode(rt, std::move(ids), std::move(batch), *ctx.model_decode_barrier,
                          fused_greedy);
        }
        const auto hc = input.target_hidden_states.value_or(Tensor{});
        auto *mtp_graph_barrier = ctx.mtp_graph_barrier;
        const bool short_verify = short_verify_layout(batch.query_offsets, ids->numel());
        const bool packed_target = cfg_.classic_mtp_target && !cfg_.is_mtp && !hc && (batch.query_offsets.size() == 2 || batch.query_offsets.size() == 4) && short_verify;
        const bool direct_hc = input.mtp_graph_direct_commit && packed_target && input.return_draft_hidden_states && !input.rollback_v4_cache;
        const bool short_target = !cfg_.is_mtp && !hc && (input.mtp_scratch_step || direct_hc || (input.rollback_v4_cache && short_verify));
        const bool draft_seed_graph = input.mtp_graph_draft_seed && cfg_.is_mtp && hc && input.return_draft_hidden_states && !input.sample_all_positions && !input.rollback_v4_cache && batch.query_offsets.size() == 2 && short_verify;
        // The K1 proposal already wrote its anchor. Repair and the next seed
        // form one two-row Draft span with only the last prediction sampled.
        const bool short_draft = cfg_.is_mtp && hc && !input.rollback_v4_cache &&
            batch.query_offsets.size() == 2 &&
            (ids->numel() == 1 || (k1_graph_seed_ && ids->numel() == 2 && !input.sample_all_positions &&
                                  batch.query_offsets.front() == 0 && batch.query_offsets.back() == 2));
        const bool scratch_hc = input.mtp_scratch_step && input.return_draft_hidden_states &&
            short_target && !input.rollback_v4_cache;
        const bool owned_hc = scratch_hc || direct_hc || draft_seed_graph;
        // Eligibility must agree across ranks before any capture or collective.
        const bool local_capture = (!input.return_draft_hidden_states || owned_hc) && (short_target || short_draft || draft_seed_graph);
        bool capture_mtp = false;
        if (mtp_graph_barrier && input.mtp_graph_direct_commit && batch.query_offsets.size() == 2) {
            // Choose the same C128 graph on every rank, together with Graph eligibility.
            const unsigned common = mtp_graph_barrier->common_decode_slots(
                (local_capture ? 1u : 0u) | (direct_hc && batch.c128_no_store ? 2u : 0u));
            capture_mtp = (common & 1u) != 0;
            batch.c128_state_only = capture_mtp && (common & 2u) != 0;
        } else if (mtp_graph_barrier) {
            capture_mtp = mtp_graph_barrier->all(local_capture);
        }
        // Both arms share the same immutable routes. No multi-row Fill is recorded.
        batch.packed_shared_constants = capture_mtp && packed_target && (input.rollback_v4_cache || direct_hc) && packed_shared_constants_prepared_ && packed_shared_constants_enabled();
        if (capture_mtp && draft_seed_graph && draft_seed_constants_prepared_) {
            const auto *mode = std::getenv("INFINILM_MTP_GRAPH_K3_DRAFT_SEED_CONSTANTS");
            // The existing plan key compares this bit before replaying a graph.
            batch.packed_shared_constants = mode && std::string_view(mode) == "1";
        }
        if (input.mtp_graph_direct_commit && (!direct_hc || !batch.packed_shared_constants)) {
            throw std::runtime_error("Direct MTP requires the prepared Target Graph path");
        }
        batch.packed_collectives = capture_mtp && direct_hc && batch.query_offsets.size() == 4 && env_enabled("INFINILM_MTP_GRAPH_C3_PACKED_TARGET_COLLECTIVES");
        constexpr std::array<size_t, 4> c3_q4_offsets{0, 4, 8, 12};
        batch.packed_mhc_pre = capture_mtp && direct_hc && rt.device().type() == runtime::Device::Type::kHygon && env_enabled("INFINILM_MTP_GRAPH_C3_PACKED_MHC_PRE") && std::equal(batch.query_offsets.begin(), batch.query_offsets.end(), c3_q4_offsets.begin(), c3_q4_offsets.end());
        if (input.mtp_graph_draft_seed && !capture_mtp) {
            throw std::runtime_error("Classic NextN Graph Draft seed requires a device Graph with owned Draft HC");
        }
        if (input.rollback_v4_cache) {
            if (!ctx.mtp_snapshot_barrier)
                throw std::invalid_argument("Classic NextN verify requires a snapshot rendezvous");
            PagedCacheSnapshot snapshot(rt,caches_,writes,
                reuse_snapshot_buffer_ ? &snapshot_backup_ : nullptr);
            // A rank that cannot allocate its snapshot must fail before its peers
            // enter the verify collectives.
            ctx.mtp_snapshot_barrier->wait();
            auto output = capture_mtp
                ? decode(rt, std::move(ids), std::move(batch), *mtp_graph_barrier, false, hc, true)
                : forward_logits(rt, ids, batch, hc, input.return_draft_hidden_states);
            snapshot.rollback();
            return output;
        }
        if (capture_mtp) {
            const auto table_width = batch.page_table->size(1);
            batch.short_c4_index = direct_hc && batch.query_offsets.size() == 2 && rt.device().type() == runtime::Device::Type::kHygon && table_width > 0 && table_width <= kC4IndexMaxUnscoredPages && env_enabled("INFINILM_MTP_GRAPH_SHORT_C4_INDEX");
            batch.share_flashmla_schedule = input.mtp_share_flashmla_schedule && direct_hc && batch.query_offsets.size() == 2 && ids->numel() == 4 && rt.device().type() == runtime::Device::Type::kHygon;
            return decode(rt, std::move(ids), std::move(batch), *mtp_graph_barrier,
                          draft_seed_graph, hc, true, owned_hc, direct_hc, draft_seed_graph);
        }
        const bool sampled_logits_only = input.sampled_logits_only && !input.sample_all_positions
            && !cfg_.is_mtp && !input.return_draft_hidden_states && !input.mtp_scratch_step;
        return forward_logits(rt, ids, batch, hc, input.return_draft_hidden_states,
                              input.mtp_last_logits, nullptr, sampled_logits_only);
    }

    bool owns_paged_decode_graph() const override {
        return !cfg_.is_mtp;
    }

    void clone_v4_cache_pages(
        const std::vector<std::pair<size_t, size_t>> &copies,
        global_state::ForwardContext &ctx, engine::RankBarrier &barrier) const override {
        const auto *paged = dynamic_cast<const cache::PagedKVCacheConfig *>(cache_config_.get());
        if (cfg_.is_mtp || !paged || paged->block_size() != kV4PageSize) {
            throw std::invalid_argument("V4 cache page clone requires a paged DeepSeek V4 Target");
        }
        clone_cache_pages(ctx.require_runtime(), caches_, paged->num_blocks(), copies, barrier);
    }

    void reset_cache(const cache::CacheConfig *requested,global_state::ForwardContext &ctx,
                     backends::AttentionBackend backend,const Rank &rank) override {
        auto &rt = ctx.require_runtime();
        cache::validate_cache_runtime(rt,rank);
        cache::validate_backend_cache_config(requested,backend);
        decode_plans_ = {};
        mtp_decode_plans_ = {};
        c128_state_only_decode_plans_ = {};
        draft_seed_decode_plans_ = {};
        snapshot_backup_.reset();
        if (!requested) {
            caches_.clear();
            cache_config_.reset();
            swa_rope_.reset();
            compressed_rope_.reset();
            return;
        }
        const auto *paged = dynamic_cast<const cache::PagedKVCacheConfig*>(requested);
        if (!paged || paged->block_size()!=kV4PageSize || !paged->num_blocks())
            throw std::invalid_argument("DeepSeek V4 requires 256-token cache pages");
        const auto rows = std::min(cfg_.max_positions,paged->num_blocks()*kV4PageSize);
        const auto &json = model_config_->get_config_json();
        auto swa = make_rotary_frequencies(rt,{64,rows,model_config_->get<double>("rope_theta")},true);
        const auto &scaling = json.at("rope_scaling");
        auto compressed = make_rotary_frequencies(rt,{64,rows,model_config_->get<double>("compress_rope_theta"),
            scaling.at("original_max_position_embeddings").get<size_t>(),scaling.at("factor").get<double>(),
            scaling.at("beta_fast").get<double>(),scaling.at("beta_slow").get<double>()},true);
        std::vector<LayerPagedCache> pending;
        pending.reserve(cfg_.ratios.size());
        for (auto ratio : cfg_.ratios)
            pending.push_back(allocate_layer_paged_cache(
                rt, paged->num_blocks(), paged->num_blocks(), ratio, 128, cfg_.classic_mtp_target));
        caches_ = std::move(pending);
        swa_rope_ = std::move(swa);
        compressed_rope_ = std::move(compressed);
        cache_config_ = requested->unique_copy();
    }
private:
    struct DecodePlan {
        Tensor ids;
        BatchMetadata batch;
        Tensor target_hc;
        Tensor logits;
        Tensor hidden_states;
        Tensor sampled_ids;
        bool greedy;
        bool return_hc;
        std::array<bool, 5> short_kernel_modes;
        std::shared_ptr<execution::graph::Graph> graph;
        size_t packed_mhc_calls = 0;
        size_t flashmla_schedules_prepared = 0;
    };

    static Tensor copy_graph_tensor(Runtime &rt, const Tensor &source) {
        if (!source) {
            return {};
        }
        auto stable = Tensor::empty(rt, source->shape(), source->dtype());
        stable->copy_from(rt, source);
        return stable;
    }

    Tensor gather_packed_logits(Runtime &rt, const Tensor &hidden) const {
        const auto rows = hidden->size(0);
        const auto local_vocab = cfg_.vocab / kV4TensorParallelSize;
        auto local = Tensor::empty(rt, {rows, local_vocab}, hidden->dtype());
        const auto copy_logits = [&](const Tensor &source, const Tensor &destination) {
            execution::submit<infini::ops::Copy>(
                rt, execution::RecordingPolicy::DeviceOnly,
                infini::ops::Copy::DefaultConfig(rt.device().type()), source, false, destination);
        };
        for (size_t row = 0; row < rows; ++row) {
            auto input = hidden->narrow({{0, row, 1}});
            copy_logits(head_->forward(rt, input), local->narrow({{0, row, 1}}));
        }

        // AllGather writes [rank, row, local_vocab]. Restore the row-major token order.
        auto gathered = Tensor::empty(rt, {kV4TensorParallelSize, rows, local_vocab}, local->dtype());
        engine::distributed::allgather_(rt, gathered, local, comm_);
        auto output = Tensor::empty(rt, {rows, kV4TensorParallelSize, local_vocab}, local->dtype());
        copy_logits(gathered->permute({1, 0, 2}), output);
        return output->view({1, rows, cfg_.vocab});
    }

    Output forward_logits(Runtime &rt,const Tensor &ids,const BatchMetadata &batch,
                          const Tensor &target_hc = {},bool return_hc = false,
                          bool last_logits = false, size_t *packed_mhc_calls = nullptr,
                          bool sampled_logits_only = false) const {
        auto [hidden,hc] = backbone_->forward(
            rt,ids,batch,caches_,swa_rope_,compressed_rope_,target_hc,return_hc,false,packed_mhc_calls);
        // A per-token projection retains the reference's M=1 accumulation boundary
        // and leaves full packed HC available to seed or repair Draft.
        if (batch.packed_collectives && !last_logits) {
            return Output{gather_packed_logits(rt, hidden), std::move(hc), std::nullopt, {}};
        }
        const auto requests = batch.query_offsets.size() - 1;
        const bool select_last = (sampled_logits_only || last_logits) && requests < ids->numel();
        const auto rows = select_last ? requests : ids->numel();
        Tensor output;
        if (batch.packed_decode_collectives) {
            // Keep the request-local M=1 LM head projection. Gather the
            // rank-local results together, then restore request-major logits.
            auto local = project_requests(rt, batch.query_offsets, [&](const RequestRows &part) {
                auto input = part.slice(hidden);
                return head_->forward(rt, input);
            });
            const auto local_vocab = local->size(1);
            auto gathered = Tensor::empty(rt, {cfg_.vocab / local_vocab, rows, local_vocab},
                                          local->dtype());
            engine::distributed::allgather_(rt, gathered, local, comm_);
            output = gathered->permute({1, 0, 2})->contiguous(rt)->view({rows, cfg_.vocab});
        } else {
            output = Tensor::empty(rt,{rows,cfg_.vocab},hidden->dtype());
            for (size_t out_row=0;out_row<rows;++out_row) {
                const auto row = select_last ? batch.query_offsets[out_row + 1] - 1 : out_row;
                auto input = hidden->narrow({{0,row,1}});
                auto local = head_->forward(rt,input);
                engine::distributed::allgather_(rt,output->narrow({{0,out_row,1}}),local,comm_);
            }
        }
        Output result{output->view({1,rows,cfg_.vocab}),std::move(hc),std::nullopt,{}};
        result.mtp_last_logits_used = last_logits && select_last;
        return result;
    }

    Output decode(Runtime &rt, Tensor ids, BatchMetadata batch, engine::RankBarrier &barrier,
                  bool greedy, const Tensor &target_hc = {}, bool classic_mtp = false,
                  bool return_hc = false, bool packed_direct = false, bool draft_seed_graph = false) const {
        auto &plans = batch.c128_state_only ? c128_state_only_decode_plans_
                                            : (draft_seed_graph ? draft_seed_decode_plans_.at(ids->numel() - 2)
                                                                : (classic_mtp ? mtp_decode_plans_ : decode_plans_));
        auto &recent = plans.front();
        auto &previous = plans.back();
        const auto incoming = batch.tensors();
        const bool m12_hygon = ids->numel() == 12 && rt.device().type() == runtime::Device::Type::kHygon;
        const std::array short_kernel_modes{
            env_enabled("INFINIOPS_V4_C128_POOL_SHORT"),
            env_enabled("INFINI_HYGON_MOE_Q4_GROUP"),
            env_enabled("INFINIOPS_V4_I8_PROJECTION_Q4"),
            m12_hygon && env_enabled("INFINI_HYGON_MOE_M12_GROUP"),
            m12_hygon && env_enabled("INFINIOPS_V4_I8_PROJECTION_M12"),
        };
        // Captured request-local slices use the full vector of row boundaries.
        auto matches = [&](const std::optional<DecodePlan> &plan) {
            if (!plan || plan->greedy != greedy || plan->return_hc != return_hc ||
                plan->batch.packed_decode_collectives != batch.packed_decode_collectives ||
                plan->batch.direct_decode_wo_a != batch.direct_decode_wo_a ||
                plan->batch.packed_moe_gate != batch.packed_moe_gate ||
                plan->batch.packed_compressor_projection != batch.packed_compressor_projection ||
                (batch.direct_decode_wo_a && batch.packed_decode_mla &&
                 plan->batch.uniform_decode_positions != batch.uniform_decode_positions) ||
                plan->batch.packed_decode_mla != batch.packed_decode_mla ||
                plan->batch.partial_decode_mla_rows != batch.partial_decode_mla_rows ||
                plan->batch.c4_sequential != batch.c4_sequential ||
                plan->batch.c4_query_begin != batch.c4_query_begin ||
                plan->batch.query_offsets != batch.query_offsets ||
                plan->batch.packed_shared_constants != batch.packed_shared_constants ||
                plan->batch.short_c4_index != batch.short_c4_index ||
                plan->batch.packed_collectives != batch.packed_collectives ||
                plan->batch.packed_mhc_pre != batch.packed_mhc_pre ||
                plan->batch.batched_decode_mhc_pre != batch.batched_decode_mhc_pre ||
                plan->batch.share_flashmla_schedule != batch.share_flashmla_schedule ||
                plan->batch.reuse_c4_flashmla_schedule != batch.reuse_c4_flashmla_schedule ||
                plan->short_kernel_modes != short_kernel_modes ||
                ids->shape() != plan->ids->shape() ||
                ids->dtype() != plan->ids->dtype()) {
                return false;
            }
            if (classic_mtp) {
                if (static_cast<bool>(plan->target_hc) != static_cast<bool>(target_hc)) {
                    return false;
                }
                if (target_hc && (plan->target_hc->shape() != target_hc->shape() ||
                    plan->target_hc->dtype() != target_hc->dtype())) {
                    return false;
                }
            }
            const auto held = plan->batch.tensors();
            for (size_t i = 0; i < held.size(); ++i) {
                if ((*held[i])->shape() != (*incoming[i])->shape() ||
                    (*held[i])->dtype() != (*incoming[i])->dtype()) {
                    return false;
                }
            }
            return true;
        };

        // Collectives must replay graphs from the same capture on every rank.
        // All ranks agree on the slot before moving it or issuing a capture.
        const unsigned local_slots = (matches(recent) ? 1u : 0u) | (matches(previous) ? 2u : 0u);
        const unsigned common_slots = barrier.common_decode_slots(local_slots);
        if (common_slots) {
            if (!(common_slots & 1u)) {
                recent.swap(previous);
            }
            recent->ids->copy_from(rt, ids);
            const auto held = recent->batch.tensors();
            for (size_t i = 0; i < held.size(); ++i) {
                Tensor dest = *held[i];
                dest->copy_from(rt, *incoming[i]);
            }
            if (target_hc) {
                recent->target_hc->copy_from(rt, target_hc);
            }
            barrier.wait();
        } else {
            // Drop the least recently used graph before pinning new allocations.
            previous.reset();
            // Keep the recorded pointers independent of caller-owned inputs.
            ids = copy_graph_tensor(rt, ids);
            batch.position = copy_graph_tensor(rt, batch.position);
            // All layers use the same constant shared-expert route. Initialize it before
            // capture to keep the Hygon Fill provider out of the decode graph.
            if (!classic_mtp && batch.packed_decode && ids->numel() > 1) {
                batch.shared_expert_route = make_shared_expert_route(rt, ids->numel());
            }
            DecodePlan candidate{std::move(ids), std::move(batch),
                                 copy_graph_tensor(rt, target_hc), {}, {}, {}, greedy, return_hc,
                                 short_kernel_modes, {}};
            barrier.wait();
            rt.startGraphRecording(true);
            try {
                if (candidate.batch.share_flashmla_schedule) {
                    auto &stable = candidate.batch;
                    const auto swa_width = stable.swa_indices->size(1);
                    for (const auto ratio : cfg_.ratios) {
                        auto &schedule = stable.flashmla_schedules[flashmla_schedule_index(ratio)];
                        if (schedule) {
                            continue;
                        }
                        const auto extra_lengths = ratio == 4   ? stable.c4_sparse_lengths
                                                 : ratio == 128 ? stable.c128_lengths
                                                                : Tensor{};
                        const auto extra_width = ratio == 4   ? kC4IndexTopK
                                               : ratio == 128 ? stable.c128_indices->size(1)
                                                              : 0;
                        schedule = prepare_sparse_mla_schedule(
                            rt, stable.swa_lengths, swa_width, extra_lengths, extra_width);
                        ++candidate.flashmla_schedules_prepared;
                    }
                }
                if (candidate.batch.reuse_c4_flashmla_schedule) {
                    auto &stable = candidate.batch;
                    // C4 keeps one-query partitions. Every C4 layer sees the same
                    // lengths, so each request can reuse its own scheduler output.
                    static_assert(kC4IndexTopK == C4SequentialIndex::kMaxCandidates);
                    const auto swa_width = stable.swa_indices->size(1);
                    stable.c4_flashmla_schedules.reserve(stable.swa_lengths->size(0));
                    for_each_request(stable.query_offsets, [&](const RequestRows &part) {
                        stable.c4_flashmla_schedules.push_back(prepare_sparse_mla_schedule(
                            rt, part.slice(stable.swa_lengths), swa_width,
                            part.slice(stable.c4_sparse_lengths), kC4IndexTopK));
                    });
                }
                auto recorded = forward_logits(
                    rt, candidate.ids, candidate.batch, candidate.target_hc, return_hc, draft_seed_graph,
                    &candidate.packed_mhc_calls);
                candidate.logits = std::move(recorded.logits);
                candidate.hidden_states = std::move(recorded.hidden_states);
                if (greedy && sampling_rank_) {
                    candidate.sampled_ids = Tensor::empty(rt, {1}, DataType::kInt64);
                    execution::submit<infini::ops::Argmax>(
                        rt, execution::RecordingPolicy::DeviceOnly,
                        infini::ops::Argmax::DefaultConfig(rt.device().type()),
                        candidate.logits->view({cfg_.vocab}), std::optional<int64_t>{}, false,
                        candidate.sampled_ids->view({}));
                }
                // No rank may allocate or synchronously upload descriptor
                // metadata while a peer is capturing on DTK.
                barrier.wait();
                candidate.graph = rt.stopGraphRecording();
                barrier.wait();
            } catch (...) {
                rt.cancelGraphRecording();
                throw;
            }
            previous = std::move(recent);
            recent = std::move(candidate);
        }
        auto report = recent->graph->execution_report();
        if (packed_direct && !barrier.all(report.device_graph_segments == 1 && report.host_segments == 0)) {
            throw std::runtime_error("Packed direct MTP requires one device Graph on every TP rank");
        }
        recent->graph->run();
        if (!classic_mtp && sampling_rank_ && env_enabled("INFINILM_DSV4_GRAPH_PLAN_LOG")) {
            std::fprintf(stderr, "DSV4_DECODE_PLAN %s rows=%zu slots=%zu operators=%zu\n",
                         common_slots ? "hit" : "capture", recent->ids->numel(),
                         plans.size(), report.operator_count);
        }
        // Publish a new allocation from the recording worker's stream. Its
        // return-HC completion fence runs before the Draft consumes this tensor.
        auto hidden_states = return_hc ? copy_graph_tensor(rt, recent->hidden_states) : Tensor{};
        if (classic_mtp && env_enabled("INFINILM_MTP_GRAPH_PLAN_RECEIPT")) {
            report.mtp_graph_plan = common_slots ? "hit" : "capture";
            if (packed_direct) {
                report.mtp_packed_mhc_calls = recent->packed_mhc_calls;
            }
            report.mtp_c128_state_only = recent->batch.c128_state_only;
            report.mtp_flashmla_schedules_prepared = recent->flashmla_schedules_prepared;
        }
        Output output{recent->logits, std::move(hidden_states), std::move(report), recent->sampled_ids};
        output.mtp_last_logits_used = draft_seed_graph;
        return output;
    }

    V4Config cfg_;
    std::shared_ptr<Backbone> backbone_;
    std::shared_ptr<ColumnLinear> head_;
    infinicclComm_t comm_;
    const bool sampling_rank_;
    std::vector<LayerPagedCache> caches_;
    bool reuse_snapshot_buffer_ = false;
    bool packed_shared_constants_prepared_ = false;
    bool draft_seed_constants_prepared_ = false;
    bool k1_graph_seed_ = false;
    // The rank worker serializes verify calls; rollback synchronizes before
    // the next request can reuse or resize this device buffer.
    mutable Tensor snapshot_backup_;
    Tensor swa_rope_, compressed_rope_;
    // These graphs hold pointers to this model instance's current KV cache.
    // The two most recent shapes cover both sides of adjacent page boundaries.
    mutable std::array<std::optional<DecodePlan>, 2> decode_plans_;
    mutable std::array<std::optional<DecodePlan>, 2> mtp_decode_plans_;
    // State-only and publishing graphs have different operator lists.
    mutable std::array<std::optional<DecodePlan>, 2> c128_state_only_decode_plans_;
    // q2, q3 and q4 repair plans do not evict the proposal plans or one another.
    mutable std::array<std::array<std::optional<DecodePlan>, 2>, 3> draft_seed_decode_plans_;
};

} // namespace

INFINILM_REGISTER_EXPLICIT_CAUSAL_LM_MODEL(deepseek_v4, DeepseekV4ForCausalLM,
                                           create_deepseek_v4_model_config);
INFINILM_REGISTER_EXPLICIT_CAUSAL_LM_MODEL(deepseek_v4_mtp, DeepseekV4ForCausalLM,
                                           create_deepseek_v4_mtp_model_config);
} // namespace infinilm::models::deepseek_v4
