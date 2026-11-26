#include "videonsa_for_conditional_generation.hpp"
#include "../../execution/tensor_expressions.hpp"
#include "../../global_state/global_state.hpp"
#include "../../weights/weights.hpp"
#include "../models_registry.hpp"
#include "runtime/runtime.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdlib>
#include <string_view>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

namespace infinilm::models::videonsa {

namespace {

std::shared_ptr<infinilm::config::ModelConfig> text_config_from(std::shared_ptr<infinilm::config::ModelConfig> model_config) {
    nlohmann::json &config_json = model_config->get_config_json();
    nlohmann::json text_config_json = config_json.contains("text_config") ? config_json["text_config"] : config_json;
    text_config_json["model_type"] = "videonsa";
    if (!text_config_json.contains("torch_dtype") && config_json.contains("torch_dtype")) {
        text_config_json["torch_dtype"] = config_json["torch_dtype"];
    }
    if (text_config_json["num_attention_heads"].get<int64_t>() <= 0) {
        throw std::invalid_argument("VideoNSA requires positive attention heads");
    }
    if (!text_config_json.contains("head_dim")) {
        text_config_json["head_dim"] = text_config_json["hidden_size"].get<size_t>() / text_config_json["num_attention_heads"].get<size_t>();
    }
    if (!text_config_json.contains("attention_bias")) {
        text_config_json["attention_bias"] = true;
    }
    return std::make_shared<infinilm::config::ModelConfig>(text_config_json);
}

// Metadata downloads use the execution owner, never the caller's TLS stream.
infinilm::runtime::Tensor host_metadata(infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &input) {
    const auto cpu = infinilm::runtime::Device{infinilm::runtime::Device::Type::kCpu};
    if (input->device() == cpu) {
        return input;
    }
    if (input->device() != runtime.device()) {
        throw std::invalid_argument("VideoNSA metadata belongs to another device");
    }
    auto dense = input->contiguous(runtime);
    auto storage = std::make_shared<std::vector<std::byte>>(dense->nbytes());
    runtime.memcpyD2H(storage->data(), dense->data(), storage->size());
    return infinilm::runtime::Tensor::from_blob(storage->data(), dense->shape(), dense->dtype(), cpu, storage);
}

} // namespace

VideoNSAForConditionalGeneration::VideoNSAForConditionalGeneration(infinilm::runtime::Runtime &runtime,
                                                                   std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                                                   const engine::distributed::RankInfo &rank,
                                                                   backends::AttentionBackend attention_backend, const layers::moe::EPConfig &ep) {
    models::validate_runtime_construction(model_config, runtime, rank);
    try {
        model_config_ = model_config;
        auto text_config = text_config_from(model_config);
        const size_t hidden_size = text_config->get<size_t>("hidden_size");
        const size_t vocab_size = text_config->get<size_t>("vocab_size");
        const auto &dtype{text_config->get_dtype()};

        model_ = this->bind_child("model", std::make_shared<std::remove_reference_t<decltype(*model_)>>(runtime, text_config, rank, attention_backend, ep));
        visual_ = this->bind_child("visual", std::make_shared<std::remove_reference_t<decltype(*visual_)>>(runtime, model_config->get_config_json()["vision_config"], dtype));
        lm_head_ = this->bind_child("lm_head", std::make_shared<std::remove_reference_t<decltype(*lm_head_)>>(runtime, hidden_size, vocab_size, false, dtype));

    } catch (...) {
        try {
            runtime.syncStream();
        } catch (...) { std::terminate(); }
        throw;
    }
}

void VideoNSAForConditionalGeneration::replace_embeddings(infinilm::runtime::Runtime &runtime,
                                                          infinilm::runtime::Tensor inputs_embeds,
                                                          const infinilm::runtime::Tensor &vision_hidden,
                                                          const infinilm::runtime::Tensor &image_bound) const {
    auto bounds_cpu = host_metadata(runtime, image_bound);
    auto out_slice = inputs_embeds->squeeze(0);
    auto bound_slice = bounds_cpu->squeeze(0);
    auto bound_count = bound_slice->size(0);
    size_t vision_offset = 0;
    for (size_t i = 0; i < bound_count; ++i) {
        auto bound = bound_slice->narrow({{0, i, 1}});
        auto bound_ptr = reinterpret_cast<const int64_t *>(bound->data());
        auto start = bound_ptr[0];
        auto end = bound_ptr[1];
        if (end <= start) {
            continue;
        }
        const size_t len = static_cast<size_t>(end - start);
        auto patch_embed = vision_hidden->narrow({{0, vision_offset, len}});
        out_slice->narrow({{0, size_t(start), len}})->copy_from(runtime, patch_embed);
        vision_offset += len;
    }
}

struct VideoNSAForConditionalGeneration::RequestGraphPlan {
    using Tensor = infinilm::runtime::Tensor;
    using Runtime = infinilm::runtime::Runtime;
    using Attention = global_state::AttentionMetadata;
    static constexpr auto inputs = std::array{
        &Attention::past_sequence_lengths, &Attention::total_sequence_lengths,
        &Attention::input_offsets, &Attention::cu_seqlens,
        &Attention::block_tables, &Attention::slot_mapping};

    Runtime *runtime{};
    std::uint64_t epoch{}, worker{};
    bool valid{true};
    std::optional<std::vector<size_t>> visual_ranges;
    std::vector<VideoNSAAttention::CompressionGraphBinding> compression;
    std::vector<Tensor> backing;
    Attention staged;
    Tensor embeds, positions, logits;
    std::shared_ptr<infinilm::execution::graph::Graph> graph;

    static bool same(const Tensor &a, const Tensor &b) {
        return a && b && a->shape() == b->shape() && a->dtype() == b->dtype()
               && a->device() == b->device();
    }
    bool matches(const Tensor &e, const Tensor &p, const global_state::ForwardContext &ctx) const {
        const auto &proof = *ctx.nsa_owner;
        if (!valid || runtime != ctx.runtime || epoch != proof.owner.pool_epoch || worker != proof.worker
            || visual_ranges != ctx.mm_metadata.visual_token_ranges
            || !same(embeds, e) || !same(positions, p) || backing.size() != ctx.kv_cache_vec.size()) {
            return false;
        }
        for (size_t i = 0; i < backing.size(); ++i) {
            const auto &now = ctx.kv_cache_vec[i];
            if (!same(backing[i], now) || backing[i]->data() != now->data()
                || backing[i]->strides() != now->strides()) {
                return false;
            }
        }
        const auto &current = ctx.attn_metadata;
        if (staged.max_query_length != current.max_query_length
            || staged.max_sequence_length != current.max_sequence_length
            || staged.logical_kv_upper_bound != current.logical_kv_upper_bound
            || staged.first_past_sequence_length != current.first_past_sequence_length
            || staged.first_total_sequence_length != current.first_total_sequence_length) {
            return false;
        }
        for (auto member : inputs) {
            const auto &old = staged.*member;
            const auto &now = current.*member;
            if (old.has_value() != now.has_value() || (old && !same(*old, *now))) {
                return false;
            }
        }
        return true;
    }
    static Tensor allocate_like(Runtime &rt, const Tensor &source) {
        if (!source || source->device() != rt.device()) {
            throw std::invalid_argument("VideoNSA graph staging requires same-device tensors");
        }
        return Tensor::empty(rt, source->shape(), source->dtype());
    }
    void upload(const Tensor &e, const Tensor &p, const Attention &current) {
        embeds->copy_from(*runtime, e);
        positions->copy_from(*runtime, p);
        for (auto member : inputs) {
            if (const auto &src = current.*member) {
                (staged.*member).value()->copy_from(*runtime, *src);
            }
        }
    }
};

VideoNSAForConditionalGeneration::Output VideoNSAForConditionalGeneration::run_cached_graph(
    const infinilm::runtime::Tensor &embeds, const infinilm::runtime::Tensor &positions,
    global_state::ForwardContext &ctx, const Input *decode_input) const {
    auto &rt = ctx.require_runtime();
    using CaptureState = infinilm::execution::graph::GraphManager::CaptureState;
    if (rt.graphCaptureState() != CaptureState::kInactive) {
        throw std::invalid_argument("VideoNSA prefill plan requires an idle graph recorder");
    }
    // Owner preflight has already validated the new request and all raw
    // layer views. The plan retains storage, never the prior request proof.
    const bool decode = decode_input != nullptr;
    std::vector<VideoNSAAttention::CompressionGraphBinding> bindings;
    if (decode) {
        const auto collect = [&](const auto &self, const infinilm::weights::ModelWeights *module) -> void {
            if (const auto *attn = dynamic_cast<const VideoNSAAttention *>(module)) {
                bindings.push_back(attn->compression_graph_binding());
            }
            for (const auto &[name, child] : module->children()) {
                (void)name;
                self(self, child.get());
            }
        };
        collect(collect, this);
        if (bindings.empty()) {
            throw std::logic_error("VideoNSA decode graph needs attention layers");
        }
    }
    const auto incremental = std::count_if(bindings.begin(), bindings.end(),
                                           [](const auto &entry) { return std::get<1>(entry); });
    const bool last_only = decode && static_cast<size_t>(incremental) == bindings.size();
    auto &slot = decode ? decode_graphs_[last_only ? 1 : 0] : prefill_graph_;
    const bool hit = slot && slot->compression == bindings && slot->matches(embeds, positions, ctx);
    if (!hit) {
        slot.reset();  // bounded owner-stream plans, no stale backing reuse.
        auto next = std::make_shared<RequestGraphPlan>();
        next->runtime = &rt;
        next->epoch = ctx.nsa_owner->owner.pool_epoch;
        next->worker = ctx.nsa_owner->worker;
        next->visual_ranges = ctx.mm_metadata.visual_token_ranges;
        next->compression = std::move(bindings);
        next->backing = ctx.kv_cache_vec;
        next->staged = ctx.attn_metadata;
        next->embeds = RequestGraphPlan::allocate_like(rt, embeds);
        next->positions = RequestGraphPlan::allocate_like(rt, positions);
        for (auto member : RequestGraphPlan::inputs) {
            if (auto &staged = next->staged.*member) {
                staged = RequestGraphPlan::allocate_like(rt, *staged);
            }
        }
        next->upload(embeds, positions, ctx.attn_metadata);
        auto record_ctx = ctx;
        record_ctx.attn_metadata = next->staged;
        rt.startGraphRecording(true);
        try {
            auto hidden = [&] {
                if (!decode) {
                    return model_->forward_embeds(next->embeds, next->positions, record_ctx);
                }
                auto input = *decode_input;
                input.input_ids = next->embeds;
                input.position_ids = next->positions;
                input.past_sequence_lengths = next->staged.past_sequence_lengths;
                input.total_sequence_lengths = next->staged.total_sequence_lengths;
                input.input_offsets = next->staged.input_offsets;
                input.cu_seqlens = next->staged.cu_seqlens;
                input.block_tables = next->staged.block_tables;
                input.slot_mapping = next->staged.slot_mapping;
                return model_->forward(input, record_ctx);
            }();
            next->logits = lm_head_->forward(rt, hidden);
            next->graph = rt.stopGraphRecording();
        } catch (...) {
            rt.cancelGraphRecording();
            throw;
        }
        slot = std::move(next);
    } else {
        slot->upload(embeds, positions, ctx.attn_metadata);
    }
    const auto &plan = *slot;
    plan.graph->run();
    if (decode) {
        mark_decode_graph_submitted();
    }
    // The engine exposes raw logits. An older caller must never alias output
    // storage mutated by the next replay.
    auto output = RequestGraphPlan::allocate_like(rt, plan.logits);
    output->copy_from(rt, plan.logits);
    auto report = plan.graph->execution_report();
    if (decode) {
        report.videonsa_decode_plan = hit ? "hit" : "capture";
        report.videonsa_decode_compression = last_only ? "incremental" : (incremental ? "mixed" : "full");
    } else {
        report.videonsa_prefill_plan = hit ? "hit" : "capture";
    }
    return {std::move(output), {}, std::move(report)};
}

void VideoNSAForConditionalGeneration::mark_decode_graph_submitted() const {
    // Recorder submission alone was not a raw/compressed-cache write.
    // Only now may per-layer owner lanes advance; completion is still
    // promoted by the worker's existing successful stream fence.
    const auto mark = [&](const auto &self, const infinilm::weights::ModelWeights *module) -> void {
        if (const auto *attention = dynamic_cast<const VideoNSAAttention *>(module)) {
            attention->mark_paged_graph_submitted();
        }
        for (const auto &[name, child] : module->children()) {
            (void)name;
            self(self, child.get());
        }
    };
    mark(mark, this);
}

infinilm::InfinilmModel::Output VideoNSAForConditionalGeneration::forward(
    const infinilm::InfinilmModel::Input &input, global_state::ForwardContext &forward_context) const {
    auto &runtime = forward_context.require_runtime();
    runtime.assertOwnerThread();
    const auto run_text = [&](auto &&body, bool decode) -> Output {
        const auto forward_logits = [&] {
            auto hidden_states = body();
            return lm_head_->forward(runtime, hidden_states);
        };
        const auto &proof = forward_context.nsa_owner;
        if (!proof || !proof->request_graph || proof->one_tail != decode) {
            return {forward_logits()};
        }

        // Request-local capture: do not retain raw KV destinations/owner proof
        // across scheduler invocations. Vision and host ownership stay outside.
        using CaptureState = infinilm::execution::graph::GraphManager::CaptureState;
        if (runtime.graphCaptureState() != CaptureState::kInactive) {
            throw std::invalid_argument("VideoNSA requires an idle graph recorder");
        }
        runtime.startGraphRecording(true);
        std::shared_ptr<infinilm::execution::graph::Graph> graph;
        infinilm::runtime::Tensor logits;
        try {
            logits = forward_logits();
            graph = runtime.stopGraphRecording();
        } catch (...) {
            runtime.cancelGraphRecording();
            throw;
        }
        graph->run();
        if (decode) {
            mark_decode_graph_submitted();
        }
        auto report = graph->execution_report();
        return {std::move(logits), {}, std::move(report)};
    };
    if (input.pixel_values.has_value() && input.pixel_values.value().size() > 0) {
        if (!input.image_bound.has_value() || !input.tgt_sizes.has_value()) {
            throw std::runtime_error("VideoNSAForConditionalGeneration: image_bound and tgt_sizes must be provided with pixel_values");
        }
        auto input_ids = input.input_ids.value();
        auto inputs_embeds = model_->embed_tokens(input_ids, forward_context);
        auto input_offsets_cpu = host_metadata(runtime, input.input_offsets.value());
        int32_t *offsets = reinterpret_cast<int32_t *>(input_offsets_cpu->data());

        const auto &image_req_ids = forward_context.mm_metadata.image_req_ids.value();
        if (input.pixel_values->size() != image_req_ids.size() || input.image_bound->size() != image_req_ids.size() || input.tgt_sizes->size() != image_req_ids.size()) {
            throw std::runtime_error("VideoNSAForConditionalGeneration: multimodal tensor lists must match image_req_ids");
        }

        std::vector<infinilm::runtime::Tensor> pixel_tensors;
        std::vector<infinilm::runtime::Tensor> grid_tensors;
        pixel_tensors.reserve(image_req_ids.size());
        grid_tensors.reserve(image_req_ids.size());
        for (size_t media_idx = 0; media_idx < image_req_ids.size(); ++media_idx) {
            pixel_tensors.push_back(input.pixel_values.value().at(media_idx));
            grid_tensors.push_back(input.tgt_sizes.value().at(media_idx));
        }
        infinilm::runtime::Tensor batched_pixels;
        if (pixel_tensors.size() == 1) {
            batched_pixels = pixel_tensors.front();
        } else {
            batched_pixels = infinilm::execution::cat_alloc(runtime, pixel_tensors, 0);
        }
        infinilm::runtime::Tensor batched_grids;
        if (grid_tensors.size() == 1) {
            batched_grids = grid_tensors.front();
        } else {
            batched_grids = infinilm::execution::cat_alloc(runtime, grid_tensors, 0);
        }
        auto batched_vision_hidden = visual_->forward(runtime, batched_pixels, batched_grids);

        size_t vision_offset = 0;
        for (size_t media_idx = 0; media_idx < image_req_ids.size(); ++media_idx) {
            const size_t req_id = image_req_ids[media_idx];
            auto bounds_cpu = host_metadata(runtime, input.image_bound.value().at(media_idx))->squeeze(0);
            auto bound_count = bounds_cpu->size(0);
            auto bounds = reinterpret_cast<const int64_t *>(bounds_cpu->data());
            size_t vision_len = 0;
            for (size_t i = 0; i < bound_count; ++i) {
                auto start = bounds[i * 2];
                auto end = bounds[i * 2 + 1];
                if (end > start) {
                    vision_len += static_cast<size_t>(end - start);
                }
            }

            auto vision_hidden = batched_vision_hidden->narrow({{0, vision_offset, vision_len}});
            auto req_embeds = inputs_embeds->narrow({{1, size_t(offsets[req_id]), size_t(offsets[req_id + 1] - offsets[req_id])}});
            replace_embeddings(runtime, req_embeds, vision_hidden, input.image_bound.value().at(media_idx));
            vision_offset += vision_len;
        }

        const auto *reuse = std::getenv("INFINILM_VIDEONSA_PREFILL_GRAPH_CACHE");
        const auto &proof = forward_context.nsa_owner;
        const auto &positions = input.position_ids.value();
        const auto &attn = forward_context.attn_metadata;
        if (reuse && std::string_view(reuse) == "1" && proof && proof->request_graph
            && !proof->one_tail && proof->owner.rows.size() == 1
            && proof->owner.rows.front().past == 0 && inputs_embeds->ndim() == 3
            && inputs_embeds->size(0) == 1 && inputs_embeds->size(1) >= 1024
            && inputs_embeds->size(1) <= 4096
            && forward_context.mm_metadata.visual_token_ranges
            && !forward_context.mm_metadata.visual_token_ranges->empty()
            && attn.slot_mapping && attn.block_tables && attn.total_sequence_lengths) {
            return run_cached_graph(inputs_embeds, positions, forward_context);
        }
        return run_text([&] {
            return model_->forward_embeds(inputs_embeds, positions, forward_context);
        }, false);
    }

    const auto *reuse = std::getenv("INFINILM_VIDEONSA_DECODE_GRAPH_CACHE");
    const auto &proof = forward_context.nsa_owner;
    const auto &attn = forward_context.attn_metadata;
    if (reuse && std::string_view(reuse) == "1" && proof && proof->request_graph
        && proof->one_tail && proof->owner.rows.size() == 1
        && input.input_ids && input.position_ids
        && input.input_ids.value()->shape() == infinilm::runtime::Shape{1, 1}
        && attn.slot_mapping && attn.block_tables && attn.total_sequence_lengths) {
        return run_cached_graph(*input.input_ids, *input.position_ids, forward_context, &input);
    }
    return run_text([&] { return model_->forward(input, forward_context); }, true);
}

void VideoNSAForConditionalGeneration::prepare_paged_owner(global_state::ForwardContext &context) const {
    // Traverse actual Module ownership, only this model's NSA attention type.
    const auto visit = [&](const auto &self, const infinilm::weights::ModelWeights *module) -> void {
        if (const auto *attention = dynamic_cast<const VideoNSAAttention *>(module)) {
            attention->prepare_paged_owner(context);
        }
        for (const auto &[name, child] : module->children()) {
            (void)name;
            self(self, child.get());
        }
    };
    visit(visit, this);
}
void VideoNSAForConditionalGeneration::finish_paged_owner(bool completed) const {
    // Traverse actual Module ownership, only this model's NSA attention type.
    const auto visit = [&](const auto &self, const infinilm::weights::ModelWeights *module) -> void {
        if (const auto *attention = dynamic_cast<const VideoNSAAttention *>(module)) {
            attention->finish_paged_owner(completed);
        }
        for (const auto &[name, child] : module->children()) {
            (void)name;
            self(self, child.get());
        }
    };
    visit(visit, this);
}
void VideoNSAForConditionalGeneration::invalidate_paged_owner() const noexcept {
    // Failure/teardown invalidates reuse immediately, but does not release
    // the recorded backing before the owner worker's drain.
    if (prefill_graph_) {
        prefill_graph_->valid = false;
    }
    for (const auto &plan : decode_graphs_) {
        if (plan) {
            plan->valid = false;
        }
    }
    // Traverse actual Module ownership, only this model's NSA attention type.
    const auto visit = [&](const auto &self, const infinilm::weights::ModelWeights *module) -> void {
        if (const auto *attention = dynamic_cast<const VideoNSAAttention *>(module)) {
            attention->invalidate_paged_owner();
        }
        for (const auto &[name, child] : module->children()) {
            (void)name;
            self(self, child.get());
        }
    };
    visit(visit, this);
}

void VideoNSAForConditionalGeneration::reset_cache(
    const cache::CacheConfig *cache_config,
    global_state::ForwardContext &forward_context,
    backends::AttentionBackend attention_backend,
    const engine::distributed::RankInfo &rank) {
    cache::validate_backend_cache_config(cache_config, attention_backend);
    // This runs on the worker after its existing owner drain.
    prefill_graph_.reset();
    for (auto &plan : decode_graphs_) {
        plan.reset();
    }
    if (nullptr == cache_config) {
        InfinilmModel::reset_cache(nullptr, forward_context, attention_backend, rank);
        return;
    }
    auto &runtime = forward_context.require_runtime();
    cache::validate_cache_runtime(runtime, rank);
    cache_config_ = cache_config->unique_copy();

    auto text_config = text_config_from(model_config_);
    auto &kv_cache_vec = forward_context.kv_cache_vec;
    kv_cache_vec.clear();
    kv_cache_vec = std::move(default_allocate_kv_cache_tensors(runtime, rank, cache_config, text_config, attention_backend));
}

std::shared_ptr<infinilm::config::ModelConfig> create_videonsa_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config) {
    const std::string &model_type = model_config->get<std::string>("model_type");
    if ("videonsa" != model_type) {
        throw std::runtime_error("infinilm::models::videonsa::create_videonsa_model_config: model_type is not videonsa");
    }

    nlohmann::json &config_json = model_config->get_config_json();
    if (config_json.contains("text_config")) {
        nlohmann::json &text_config_json = config_json["text_config"];
        if (text_config_json["num_attention_heads"].get<int64_t>() <= 0) {
            throw std::invalid_argument("VideoNSA requires positive attention heads");
        }
        if (!text_config_json.contains("head_dim")) {
            text_config_json["head_dim"] = text_config_json["hidden_size"].get<size_t>() / text_config_json["num_attention_heads"].get<size_t>();
        }
        if (!text_config_json.contains("attention_bias")) {
            text_config_json["attention_bias"] = true;
        }
        if (!config_json.contains("torch_dtype") && text_config_json.contains("torch_dtype")) {
            config_json["torch_dtype"] = text_config_json["torch_dtype"];
        }
    } else {
        if (model_config->get<int64_t>("num_attention_heads") <= 0) {
            throw std::invalid_argument("VideoNSA requires positive attention heads");
        }
        if (!config_json.contains("head_dim")) {
            config_json["head_dim"] = model_config->get<size_t>("hidden_size") / model_config->get<size_t>("num_attention_heads");
        }
        if (!config_json.contains("attention_bias")) {
            config_json["attention_bias"] = true;
        }
    }
    if (config_json.contains("vision_config")) {
        const auto &vision = config_json["vision_config"];
        if (vision.value("depth", 32) > 0) {
            const int width = vision.value("hidden_size", 1280), heads = vision.value("num_heads", 16);
            if (width <= 0 || heads <= 0 || width % heads != 0) {
                throw std::invalid_argument("VideoNSA vision head geometry is invalid");
            }
        }
    }
    return model_config;
}

} // namespace infinilm::models::videonsa

namespace {
INFINILM_REGISTER_EXPLICIT_CAUSAL_LM_MODEL(
    videonsa,
    infinilm::models::videonsa::VideoNSAForConditionalGeneration,
    infinilm::models::videonsa::create_videonsa_model_config);
} // namespace
