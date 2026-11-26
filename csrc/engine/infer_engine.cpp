#include "infer_engine.hpp"
#include "../cache/kv_cache.hpp"
#include "../config/config_factory.hpp"
#include "../models/deepseek_v4/paged_cache.hpp"
#include "deepseek_v4_metadata.hpp"
#include "../weights/weights.hpp"
#include "runtime/runtime.hpp"
#include "spdlog/spdlog.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <future>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <unordered_set>

namespace infinilm::engine {
namespace {

void validate_packed_direct_metadata(const RankWorker::Input &input,
                                     const cache::PagedKVCacheConfig &cache, size_t vocab_size,
                                     size_t owners) {
    using namespace infinilm::runtime;
    const auto cpu_tensor = [](const std::optional<Tensor> &value, DataType type, size_t dimensions) {
        return value && *value && (*value)->device().type() == Device::Type::kCpu
            && (*value)->dtype() == type && (*value)->is_contiguous() && (*value)->ndim() == dimensions;
    };
    const auto &ids = input.input_ids;
    const auto &offsets = input.input_offsets;
    const auto &positions = input.position_ids;
    const auto &past = input.past_sequence_lengths;
    const auto &total = input.total_sequence_lengths;
    const auto &cu = input.cu_seqlens;
    const auto &tables = input.block_tables;
    const auto &slots = input.slot_mapping;
    if (!cpu_tensor(ids, DataType::kInt64, 2) || (*ids)->size(0) != 1
        || !cpu_tensor(offsets, DataType::kInt32, 1) || (*offsets)->numel() != owners + 1
        || !cpu_tensor(positions, DataType::kInt64, 1)
        || !cpu_tensor(past, DataType::kInt32, 1) || (*past)->numel() != owners
        || !cpu_tensor(total, DataType::kInt32, 1) || (*total)->numel() != owners
        || !cpu_tensor(cu, DataType::kInt32, 1) || (*cu)->numel() != owners + 1
        || !cpu_tensor(tables, DataType::kInt32, 2) || (*tables)->size(0) != owners
        || !cpu_tensor(slots, DataType::kInt64, 1)) {
        throw std::invalid_argument("Direct MTP requires contiguous CPU metadata for each owner");
    }
    const auto rows = (*ids)->numel();
    const auto *token = reinterpret_cast<const int64_t *>((*ids)->data());
    const auto *bounds = reinterpret_cast<const int32_t *>((*offsets)->data());
    const auto *lengths = reinterpret_cast<const int32_t *>((*cu)->data());
    const auto *starts = reinterpret_cast<const int32_t *>((*past)->data());
    const auto *ends = reinterpret_cast<const int32_t *>((*total)->data());
    const auto *pages = reinterpret_cast<const int32_t *>((*tables)->data());
    const auto *pos = reinterpret_cast<const int64_t *>((*positions)->data());
    const auto *slot = reinterpret_cast<const int64_t *>((*slots)->data());
    const auto width = (*tables)->size(1);
    const auto page_size = static_cast<int64_t>(cache.block_size());
    if (rows < 2 * owners || rows > 4 * owners || (*positions)->numel() != rows || (*slots)->numel() != rows
        || bounds[0] != 0 || bounds[owners] != static_cast<int32_t>(rows) || lengths[0] != 0) {
        throw std::invalid_argument("Packed direct MTP offsets must cover all query rows");
    }

    // The scheduler proves lease ownership. Validate the effective physical page layout
    // here before submitting the irreversible write to any rank.
    std::unordered_set<int32_t> owner_pages;
    for (size_t owner = 0; owner < owners; ++owner) {
        const int64_t begin = bounds[owner], end = bounds[owner + 1], start = starts[owner];
        if (begin < 0 || end > static_cast<int64_t>(rows) || end - begin < 2 || end - begin > 4
            || start <= 0 || ends[owner] != start + end - begin
            || lengths[owner + 1] != int64_t{lengths[owner]} + ends[owner]) {
            throw std::invalid_argument("Packed direct MTP requires contiguous q2-q4 for each owner");
        }
        const auto used = static_cast<size_t>((int64_t{ends[owner]} - 1) / page_size + 1);
        if (used > width) {
            throw std::invalid_argument("Packed direct MTP page table is shorter than the query");
        }
        const auto *table = pages + owner * width;
        for (size_t column = 0; column < width; ++column) {
            const auto page = table[column];
            if (column < used) {
                if (page < 0 || static_cast<size_t>(page) >= cache.num_blocks()
                    || !owner_pages.insert(page).second) {
                    throw std::invalid_argument("Packed direct MTP requires exclusive effective pages");
                }
            } else if (page != -1) {
                throw std::invalid_argument("Packed direct MTP requires -1 page padding");
            }
        }
        for (int64_t row = begin; row < end; ++row) {
            const auto position = start + row - begin;
            if (pos[row] != position
                || slot[row] != int64_t{table[position / page_size]} * page_size + position % page_size) {
                throw std::invalid_argument("Packed direct MTP position and slot disagree with the owner table");
            }
            if (token[row] < 0 || static_cast<size_t>(token[row]) >= vocab_size) {
                throw std::invalid_argument("Packed direct MTP input token is outside the vocabulary");
            }
        }
    }
}

// Checked value issuer only: no identities/leases/requests stored globally.
// Unique across live engines/resets in this native process; never wraps.
std::uint64_t issue_nsa_pool_epoch() {
    static std::atomic<std::uint64_t> next{1};
    auto value = next.load(std::memory_order_relaxed);
    for (;;) {
        if (value == std::numeric_limits<std::uint64_t>::max()) {
            throw std::overflow_error("NSA native pool epoch exhausted");
        }
        if (next.compare_exchange_weak(value, value + 1, std::memory_order_relaxed)) {
            return value;
        }
    }
}

std::string weight_key_error(const char *kind, std::vector<std::string> keys) {
    std::sort(keys.begin(), keys.end());
    std::string result = std::string(kind) + " key(s) in state_dict: ";
    for (size_t i = 0; i < keys.size(); ++i) {
        if (i) {
            result += ", ";
        }
        result += '"' + keys[i] + '"';
    }
    return result;
}

size_t max_length_from_offsets(
    const std::optional<infinilm::runtime::Tensor> &offsets,
    const char *name) {
    if (!offsets.has_value()) {
        return 0;
    }

    auto cpu_offsets = offsets.value();
    if (cpu_offsets->device().type() != infinilm::runtime::Device::Type::kCpu) {
        cpu_offsets = cpu_offsets->to(
            infinilm::runtime::Device{infinilm::runtime::Device::Type::kCpu});
    }

    if (cpu_offsets->dtype() != infinilm::runtime::DataType::kInt32
        || cpu_offsets->shape().size() != 1
        || cpu_offsets->shape()[0] < 2) {
        throw std::invalid_argument(
            std::string(name) + " must be a one-dimensional int32 tensor with at least two entries");
    }

    const auto *values = reinterpret_cast<const int32_t *>(cpu_offsets->data());
    size_t max_length = 0;
    for (size_t i = 1; i < cpu_offsets->shape()[0]; ++i) {
        if (values[i] < values[i - 1]) {
            throw std::invalid_argument(std::string(name) + " must be nondecreasing");
        }
        max_length = std::max(
            max_length,
            static_cast<size_t>(values[i] - values[i - 1]));
    }
    return max_length;
}

std::optional<size_t> first_sequence_length(
    const std::optional<infinilm::runtime::Tensor> &lengths,
    const char *name) {
    if (!lengths.has_value()) {
        return std::nullopt;
    }

    auto cpu_lengths = lengths.value();
    if (cpu_lengths->device().type() != infinilm::runtime::Device::Type::kCpu) {
        cpu_lengths = cpu_lengths->to(
            infinilm::runtime::Device{infinilm::runtime::Device::Type::kCpu});
    }

    if (cpu_lengths->dtype() != infinilm::runtime::DataType::kInt32
        || cpu_lengths->shape().size() != 1
        || cpu_lengths->shape()[0] == 0) {
        throw std::invalid_argument(
            std::string(name) + " must be a non-empty one-dimensional int32 tensor");
    }

    const auto value = reinterpret_cast<const int32_t *>(cpu_lengths->data())[0];
    if (value < 0) {
        throw std::invalid_argument(std::string(name) + " must contain nonnegative lengths");
    }
    return static_cast<size_t>(value);
}

} // namespace

//------------------------------------------------------
// Constructor
//------------------------------------------------------
InferEngine::InferEngine(
    const std::string &config_str,
    const distributed::DistConfig &distributed_config,
    infinilm::runtime::Device::Type device_type,
    const cache::CacheConfig *cache_config,
    bool enable_graph_compiling,
    backends::AttentionBackend attention_backend,
    std::optional<infinilm::runtime::DataType> kv_cache_dtype,
    bool use_mla,
    const std::string &weight_load_mode,
    bool pre_transpose)
    : communication_group_(distributed_config, device_type),
      attention_backend_(attention_backend),
      weight_load_mode_(weight_load_mode),
      use_mla_(use_mla),
      enable_graph_compiling_(enable_graph_compiling) {
    if (weight_load_mode_ != "async" && weight_load_mode_ != "sync") {
        throw std::invalid_argument("weight_load_mode must be either 'async' or 'sync'");
    }
    if (attention_backend_ == backends::AttentionBackend::FLASH_ATTN) {
        const auto *paged_cache_config = dynamic_cast<const cache::PagedKVCacheConfig *>(cache_config);
        if (paged_cache_config == nullptr) {
            throw std::invalid_argument(
                "flash-attn requires a paged KV cache configuration");
        }
        if (paged_cache_config->num_blocks() == 0) {
            throw std::invalid_argument(
                "flash-attn requires at least one paged KV cache block");
        }
        if (paged_cache_config->block_size() == 0) {
            throw std::invalid_argument("flash-attn requires a nonzero paged KV cache block size");
        }
    }
    if (cache_config != nullptr) {
        cache_config_ = cache_config->unique_copy();
    }

    // Load model config if model_path is provided, model_path must be valid, and config.json exists
    this->model_config_ = infinilm::config::ConfigFactory::createConfig(config_str);
    classic_mtp_target_ = model_config_->get<std::string>("model_type") == "deepseek_v4" &&
        model_config_->get_or<bool>(models::deepseek_v4::kClassicMtpTargetKey, false);
    auto infinilm_config = std::make_shared<infinilm::global_state::InfinilmConfig>(
        attention_backend,
        this->model_config_,
        use_mla,
        distributed_config.moe_ep_backend,
        distributed_config.moe_ep_size,
        pre_transpose);

    // Only support offline int8 kv cache quantization in this version
    if (kv_cache_dtype.has_value()) {
        this->model_config_->set_kv_quant_scheme(kv_cache_dtype.value());
    }
    if (model_config_->get<std::string>("model_type") == "videonsa"
        && dynamic_cast<const cache::PagedKVCacheConfig *>(cache_config_.get()) != nullptr) {
        nsa_pool_epoch_ = issue_nsa_pool_epoch();
    }
    // Create one RankWorker per rank
    int local_tp_size = communication_group_.tp_size();
    barrier_ = std::make_unique<RankBarrier>((size_t)local_tp_size);
    workers_.reserve(local_tp_size);
    try {
        for (int r = 0; r < local_tp_size; ++r) {
            workers_.emplace_back(std::make_unique<RankWorker>(
                infinilm_config,
                communication_group_.get_rank_info(r),
                cache_config_ != nullptr ? cache_config_.get() : nullptr,
                barrier_.get(),
                enable_graph_compiling,
                attention_backend_, nsa_pool_epoch_));
        }

        for (auto &worker : workers_) {
            worker->wait_for_init();
        }
    } catch (...) {
        const auto error = std::current_exception();
        for (auto &worker : workers_) {
            worker->request_stop();
        }
        barrier_->abort(error);
        for (auto &worker : workers_) {
            worker->join();
        }
        std::rethrow_exception(error);
    }
    // Graphs must be compiled after weights are loaded and post-processed.
    // Quantized models may replace their linear implementations during
    // process_weights_after_loading(), so compiling here would capture stale
    // fallback operators.
}

//------------------------------------------------------
// load_param
//------------------------------------------------------
void InferEngine::load_param(const std::string &name, const infinilm::runtime::Tensor &param) {
    std::lock_guard<std::mutex> submission_lock(submission_mutex_);
    if (weight_state_ != WeightState::Loading) {
        throw std::logic_error("weights can only be loaded while the engine is Loading");
    }
    weight_load_started_ = true;
    try {
        if (loaded_weight_keys_.count(name)) {
            throw std::runtime_error(weight_key_error("Duplicate", {name}));
        }
        for (auto &worker : workers_) {
            worker->load_param(name, param);
        }
        loaded_weight_keys_.insert(name);
    } catch (...) {
        weight_state_ = WeightState::Failed;
        throw;
    }
}

void InferEngine::load_params(const std::unordered_map<std::string, infinilm::runtime::Tensor> &params, bool strict) {
    std::lock_guard<std::mutex> submission_lock(submission_mutex_);
    if (weight_state_ != WeightState::Loading) {
        throw std::logic_error("weights can only be loaded while the engine is Loading");
    }
    if (!params.empty()) {
        weight_load_started_ = true;
    }
    try {
        std::vector<std::string> duplicates;
        for (const auto &[key, _] : params) {
            if (loaded_weight_keys_.count(key)) {
                duplicates.push_back(key);
            }
        }
        if (!duplicates.empty()) {
            throw std::runtime_error(weight_key_error("Duplicate", std::move(duplicates)));
        }
        if (workers_.size() <= 1 || weight_load_mode_ == "sync") {
            for (auto &worker : workers_) {
                worker->load_params(params, strict);
            }
        } else {
            std::vector<std::future<void>> futures;
            futures.reserve(workers_.size());
            for (auto &worker : workers_) {
                futures.emplace_back(std::async(std::launch::async, [worker_ptr = worker.get(), &params, strict] {
                    worker_ptr->load_params(params, strict);
                }));
            }
            for (auto &future : futures) {
                future.get();
            }
        }
        for (const auto &[key, _] : params) {
            loaded_weight_keys_.insert(key);
        }
    } catch (...) {
        weight_state_ = WeightState::Failed;
        throw;
    }
}

//------------------------------------------------------
// load_param
//------------------------------------------------------
void InferEngine::process_weights_after_loading() {
    std::lock_guard<std::mutex> submission_lock(submission_mutex_);
    if (weight_state_ != WeightState::Loading) {
        throw std::logic_error("weights can only be finalized once from Loading");
    }
    weight_state_ = WeightState::Finalizing;
    try {
        const auto expected_keys = state_dict_keys();
        const std::unordered_set<std::string> expected(expected_keys.begin(), expected_keys.end());
        std::vector<std::string> missing;
        std::vector<std::string> unexpected;
        std::unordered_set<std::string> required;
        for (auto &worker : workers_) {
            for (const auto &key : worker->state_required_keys()) {
                required.insert(key);
            }
        }
        for (const auto &key : required) {
            if (!loaded_weight_keys_.count(key)) {
                missing.push_back(key);
            }
        }
        for (const auto &key : loaded_weight_keys_) {
            if (!expected.count(key)) {
                unexpected.push_back(key);
            }
        }
        if (!missing.empty() || !unexpected.empty()) {
            std::string message = "Error(s) in loading state_dict";
            if (!unexpected.empty()) {
                message += "\n" + weight_key_error("Unexpected", std::move(unexpected));
            }
            if (!missing.empty()) {
                message += "\n" + weight_key_error("Missing", std::move(missing));
            }
            throw std::runtime_error(message);
        }
        const auto scheme = model_config_->get_quant_scheme();
        if (scheme == infinilm::quantization::QuantScheme::AWQ_W4A16 || scheme == infinilm::quantization::QuantScheme::GPTQ_W4A16) {
            // Every worker rejects unsupported/mixed coverage before ANY stage.
            for (auto &worker : workers_) {
                worker->validate_w4_weights();
            }
            for (auto &worker : workers_) {
                worker->prepare_w4_weights();
            }
            // All prepare calls acknowledge successful owner sync/trim/seal.
            for (auto &worker : workers_) {
                worker->commit_w4_weights();
            }
        } else {
            for (auto &worker : workers_) {
                worker->process_weights_after_loading();
            }
        }
        compile_workers();
        weight_state_ = WeightState::Ready;
    } catch (...) {
        weight_state_ = WeightState::Failed;
        throw;
    }
}

//------------------------------------------------------
// state_dict
//------------------------------------------------------
std::vector<std::unordered_map<std::string, infinilm::weights::Weight>> InferEngine::state_dict() {
    std::vector<std::unordered_map<std::string, infinilm::weights::Weight>> results;
    if (0 == workers_.size()) {
        throw std::runtime_error(" Model object not found. ");
    }

    for (auto &worker : workers_) {
        results.push_back(worker->state_dict());
    }
    return results;
}

std::vector<std::string> InferEngine::state_dict_keys() {
    if (0 == workers_.size()) {
        throw std::runtime_error(" Model object not found. ");
    }
    std::vector<std::string> ordered_keys;
    std::unordered_set<std::string> seen_keys;
    for (auto &worker : workers_) {
        for (const auto &key : worker->state_dict_keys()) {
            // Preserve first-seen worker order while removing duplicate TP keys.
            if (seen_keys.emplace(key).second) {
                ordered_keys.push_back(key);
            }
        }
    }
    return ordered_keys;
}

//------------------------------------------------------
// forward
//------------------------------------------------------
infinilm::InfinilmModel::Input
InferEngine::Input::to_model_input(
    infinilm::runtime::Device device,
    ForwardContext &forward_context,
    bool snapshot_static_sequence_lengths,
    bool preserve_target_hidden_device,
    size_t logical_kv_upper_bound,
    std::shared_ptr<const nsa::Proof> nsa_owner) const {

    forward_context.nsa_owner = std::move(nsa_owner);
    auto &runtime = forward_context.require_runtime();
    if (device.type() != infinilm::runtime::Device::Type::kCpu && runtime.device() != device) {
        throw std::invalid_argument("input transfer Runtime and target device differ");
    }
    auto to_device = [&](const std::optional<infinilm::runtime::Tensor> &t)
        -> std::optional<infinilm::runtime::Tensor> {
        if (!t.has_value() || t.value()->device() == device) {
            return t;
        }
        if (t.value()->device().type() == infinilm::runtime::Device::Type::kCpu
            && device.type() != infinilm::runtime::Device::Type::kCpu) {
            auto result = infinilm::runtime::Tensor::empty(runtime, t.value()->shape(), t.value()->dtype());
            result->copy_from(runtime, t.value());
            return result;
        }
        auto source = t.value()->allocation_owner();
        if (!source || source->device() != t.value()->device()) {
            source = infinilm::runtime::context::resolveRuntime(t.value()->device());
        }
        return device.type() == infinilm::runtime::Device::Type::kCpu
                 ? t.value()->to(*source, device)
                 : t.value()->to(*source, runtime);
    };
    auto to_device_vec = [&](const std::optional<std::vector<infinilm::runtime::Tensor>> &vec)
        -> std::optional<std::vector<infinilm::runtime::Tensor>> {
        if (!vec.has_value()) {
            return vec;
        }
        std::vector<infinilm::runtime::Tensor> result;
        result.reserve(vec->size());
        for (const auto &t : vec.value()) {
            result.push_back(to_device(t).value());
        }
        return result;
    };

    const bool is_prefill = input_ids.has_value()
                         && total_sequence_lengths.has_value()
                         && input_ids.value()->numel()
                                != total_sequence_lengths.value()->numel();
    const size_t max_query_length = is_prefill ? max_length_from_offsets(input_offsets, "input_offsets") : 0;
    const size_t max_sequence_length = is_prefill ? max_length_from_offsets(cu_seqlens, "cu_seqlens") : 0;

    std::optional<size_t> first_past_sequence_length;
    std::optional<size_t> first_total_sequence_length;
    if (snapshot_static_sequence_lengths) {
        first_past_sequence_length = first_sequence_length(
            past_sequence_lengths, "past_sequence_lengths");
        first_total_sequence_length = first_sequence_length(
            total_sequence_lengths, "total_sequence_lengths");
    }

    infinilm::InfinilmModel::Input input = {
        to_device(input_ids), // @todo: on device in the future
        to_device(position_ids),
        to_device(past_sequence_lengths), // @todo: on device in the future
        to_device(total_sequence_lengths),
        to_device(input_offsets),
        to_device(cu_seqlens),
        to_device(block_tables),
        to_device(slot_mapping),
        to_device(mamba_init_state_indices),
        to_device(mamba_final_state_indices),
        to_device_vec(pixel_values),
        to_device_vec(image_bound),
        to_device_vec(tgt_sizes),
        to_device_vec(image_grid_thw),
        image_req_ids,
        visual_token_ranges,
        preserve_target_hidden_device
            ? target_hidden_states
            : to_device(target_hidden_states),
        sample_all_positions};
    if (!target_hidden_segments.empty()) {
        // The producer fence completed before this RUN. Keep the source views
        // and enqueue all copies on this Draft worker's stream before forward.
        const auto &first = target_hidden_segments.front();
        auto packed = infinilm::runtime::Tensor::empty(runtime,
            {input.input_ids.value()->numel(), first->size(1), first->size(2)}, first->dtype());
        size_t offset = 0;
        for (const auto &view : target_hidden_segments) {
            const auto rows = view->size(0);
            packed->narrow({{0, offset, rows}})->copy_from(runtime, view);
            offset += rows;
        }
        input.target_hidden_states = std::move(packed);
    }
    input.logical_kv_upper_bound = logical_kv_upper_bound;
    input.return_draft_hidden_states = return_draft_hidden_states;
    input.rollback_v4_cache = rollback_v4_cache;
    input.mtp_draft_cache_only = mtp_draft_cache_only;
    input.mtp_scratch_step = mtp_scratch_step;
    input.mtp_last_logits = mtp_last_logits;
    input.mtp_graph_direct_commit = mtp_graph_direct_commit;
    input.mtp_graph_draft_seed = mtp_graph_draft_seed;

    auto &attn_metadata = forward_context.attn_metadata;
    attn_metadata = {
        input.past_sequence_lengths,
        input.total_sequence_lengths,
        input.input_offsets,
        input.cu_seqlens,
        input.block_tables,
        input.slot_mapping,
        max_query_length,
        max_sequence_length};
    attn_metadata.first_past_sequence_length = first_past_sequence_length;
    attn_metadata.first_total_sequence_length = first_total_sequence_length;
    attn_metadata.logical_kv_upper_bound = logical_kv_upper_bound;

    forward_context.mamba_metadata = {
        input.input_offsets,
        input.mamba_init_state_indices,
        input.mamba_final_state_indices};

    forward_context.mm_metadata = {
        image_req_ids,
        visual_token_ranges};

    return input;
}

InferEngine::Output InferEngine::forward(const InferEngine::Input &original) {
    std::lock_guard<std::mutex> submission_lock(submission_mutex_);
    if (weight_state_ == WeightState::Failed
        || weight_state_ == WeightState::Finalizing
        || (weight_state_ == WeightState::Loading && weight_load_started_)) {
        throw std::logic_error("cannot execute an incomplete or failed weight load");
    }
    std::shared_ptr<const DeepSeekV4HostBatch> v4_host_batch;
    std::optional<Input> frozen_dispatch;
    const auto *paged = dynamic_cast<const cache::PagedKVCacheConfig *>(cache_config_.get());
    const auto *direct_plan = std::getenv("INFINILM_MTP_GRAPH_C3_SHARED_TARGET_PLAN");
    // This native-only path freezes the exact tensors used by the direct-write admission below.
    const bool direct_host_plan = original.mtp_graph_direct_commit && enable_graph_compiling_
        && paged && paged->max_batch_size() == 3 && direct_plan && std::string_view(direct_plan) == "1";
    const bool eager_host_batch = !enable_graph_compiling_
        && (original.mtp_cpu_metadata_shadow || original.mtp_shared_compression_plan);
    if ((direct_host_plan || eager_host_batch) && communication_group_.get_dist_config().pp_size == 1
        && original.input_ids && *original.input_ids && (*original.input_ids)->ndim() == 2
        && (*original.input_ids)->size(0) == 1) {
        const auto type = model_config_->get<std::string>("model_type");
        const bool classic = (type == "deepseek_v4" &&
                              (original.return_draft_hidden_states || original.rollback_v4_cache))
            || (type == "deepseek_v4_mtp" &&
                (original.target_hidden_states_by_rank || original.target_hidden_segments_by_rank));
        if (classic && paged && paged->block_size() == models::deepseek_v4::kV4PageSize
            && original.position_ids && original.input_offsets &&
            original.block_tables && original.slot_mapping) {
            v4_host_batch = DeepSeekV4HostBatch::prepare_short(
                *original.position_ids, *original.input_offsets, *original.block_tables,
                *original.slot_mapping, (*original.input_ids)->numel(), paged->num_blocks(),
                direct_host_plan || original.mtp_shared_compression_plan, classic_mtp_target_);
            if (v4_host_batch) {
                frozen_dispatch.emplace(original);
                frozen_dispatch->position_ids = v4_host_batch->positions();
                frozen_dispatch->input_offsets = v4_host_batch->offsets();
                frozen_dispatch->block_tables = v4_host_batch->tables();
                frozen_dispatch->slot_mapping = v4_host_batch->slots();
            }
        }
    }
    const auto &input = frozen_dispatch ? *frozen_dispatch : original;
    if (input.random_values.has_value()) {
        if (input.sample_all_positions || !input.input_offsets.has_value()
            || input.input_offsets.value()->ndim() != 1
            || input.input_offsets.value()->size(0) < 2
            || input.random_values->size() != input.input_offsets.value()->size(0) - 1) {
            throw std::invalid_argument(
                "random_values requires one value per request and last-token sampling");
        }
        for (float value : *input.random_values) {
            if (!std::isfinite(value) || value < 0.0f || value >= 1.0f) {
                throw std::invalid_argument("random_values must be finite values in [0, 1)");
            }
        }
    }
    if (input.sampling_rows.has_value()) {
        if (input.sample_all_positions || !input.input_offsets.has_value()
            || input.input_offsets.value()->ndim() != 1
            || input.input_offsets.value()->size(0) < 2
            || input.sampling_rows->size() != input.input_offsets.value()->size(0) - 1) {
            throw std::invalid_argument(
                "sampling_rows requires one value per request and last-token sampling");
        }
        for (const auto &[temperature, top_p, top_k] : *input.sampling_rows) {
            if (!std::isfinite(temperature) || temperature < 0.0f
                || !std::isfinite(top_p) || top_p < 0.0f || top_p > 1.0f
                || top_k <= 0) {
                throw std::invalid_argument("sampling_rows contains invalid parameters");
            }
        }
    }
    const auto &rank_hidden = input.target_hidden_states_by_rank;
    const auto &rank_segments = input.target_hidden_segments_by_rank;
    if (input.mtp_graph_direct_commit) {
        const auto &dist = communication_group_.get_dist_config();
        const auto owners = paged ? paged->max_batch_size() : 0;
        const auto *enabled = std::getenv(owners == 1
            ? "INFINILM_MTP_GRAPH_K3_DIRECT_COMMIT" : "INFINILM_MTP_GRAPH_C3_DIRECT_COMMIT");
        const auto [c4_ring, c128_ring] = deepseek_v4_raw_ring_rows();
        if (!enabled || std::string_view(enabled) != "1" || !enable_graph_compiling_
            || !classic_mtp_target_ || nsa_pool_epoch_ || dist.pp_size != 1
            || dist.tp_device_ids.size() != 8 || dist.moe_ep_size != 1 || !paged
            || (owners != 1 && owners != 3) || paged->block_size() != models::deepseek_v4::kV4PageSize
            || c4_ring != 16 || c128_ring != 256
            || rank_hidden || input.target_hidden_states || !input.return_draft_hidden_states
            || !input.sample_all_positions || input.top_k != 1 || input.sampling_rows || input.random_values
            || input.rollback_v4_cache || input.mtp_scratch_step || input.mtp_last_logits
            || input.mtp_draft_cache_only || input.mtp_cpu_metadata_shadow || input.mtp_shared_compression_plan) {
            throw std::invalid_argument("Packed direct MTP requires plain TP8 greedy Target Graph verification");
        }
        validate_packed_direct_metadata(input, *paged, model_config_->get<size_t>("vocab_size"), owners);
    }
    using namespace infinilm::runtime;
    const auto cpu_vector = [](const std::optional<Tensor> &value, DataType type) {
        return value && *value && (*value)->device().type() == Device::Type::kCpu
            && (*value)->dtype() == type && (*value)->is_contiguous() && (*value)->ndim() == 1;
    };
    if (rank_segments) {
        const auto &ids = input.input_ids;
        const auto &offsets = input.input_offsets;
        if (model_config_->get<std::string>("model_type") != "deepseek_v4_mtp"
            || communication_group_.get_dist_config().pp_size != 1 || nsa_pool_epoch_
            || rank_hidden || input.target_hidden_states || rank_segments->size() != workers_.size()
            || !ids || !*ids || (*ids)->ndim() != 2 || (*ids)->size(0) != 1
            || !cpu_vector(offsets, DataType::kInt32) || (*offsets)->numel() < 3
            || input.sample_all_positions || input.top_k != 1 || input.sampling_rows || input.random_values
            || input.rollback_v4_cache || input.mtp_draft_cache_only || input.mtp_scratch_step || input.mtp_last_logits
            || input.mtp_graph_direct_commit || input.mtp_graph_fresh_prefill || input.mtp_graph_draft_seed) {
            throw std::invalid_argument("Segmented rank-local HC requires packed greedy Classic V4 Draft");
        }
        const auto *bounds = reinterpret_cast<const int32_t *>((*offsets)->data());
        const auto segments = (*offsets)->numel() - 1;
        if (bounds[0] != 0 || bounds[segments] <= 0 || size_t(bounds[segments]) != (*ids)->numel()) {
            throw std::invalid_argument("Packed Draft offsets must cover all input tokens");
        }
        for (size_t segment = 0; segment < segments; ++segment) {
            if (bounds[segment + 1] <= bounds[segment]) {
                throw std::invalid_argument("Packed Draft offsets must strictly increase");
            }
        }
        const auto hidden_size = model_config_->get<size_t>("hidden_size");
        for (size_t rank = 0; rank < workers_.size(); ++rank) {
            const auto &views = (*rank_segments)[rank];
            const auto device = communication_group_.get_rank_info(rank).device;
            if (views.size() != segments) {
                throw std::invalid_argument("Each Draft rank requires one HC view per request");
            }
            for (size_t segment = 0; segment < segments; ++segment) {
                const auto &view = views[segment];
                const auto count = size_t(bounds[segment + 1] - bounds[segment]);
                if (!view || view->device() != device || view->dtype() != DataType::kBFloat16
                    || view->ndim() != 3 || view->size(0) != count
                    || view->size(1) != 4 || view->size(2) != hidden_size) {
                    throw std::invalid_argument("Packed Draft rank-local HC view does not match its input rows");
                }
            }
        }
    }
    if (input.mtp_graph_fresh_prefill) {
        // Graph-enabled engines still run this prefill eagerly. Permit only the
        // initial Target head selection or the initial output-free Draft write.
        const auto *paged = dynamic_cast<const cache::PagedKVCacheConfig *>(cache_config_.get());
        if (!enable_graph_compiling_ || communication_group_.get_dist_config().pp_size != 1
            || !paged || paged->max_batch_size() != 1
            || (input.mtp_last_logits && !classic_mtp_target_)
            || input.mtp_last_logits == input.mtp_draft_cache_only
            || input.mtp_last_logits == rank_hidden.has_value()
            || input.return_draft_hidden_states != input.mtp_last_logits
            || input.sample_all_positions != input.mtp_draft_cache_only
            || input.target_hidden_states || input.rollback_v4_cache || input.mtp_scratch_step
            || input.mtp_graph_direct_commit || input.mtp_graph_draft_seed
            || input.mtp_cpu_metadata_shadow || input.mtp_shared_compression_plan
            || input.top_k != 1 || input.random_values || input.sampling_rows) {
            throw std::invalid_argument("Classic NextN Graph prefill requires a fresh greedy Target or Draft");
        }
        const auto &ids = input.input_ids;
        const auto &offsets = input.input_offsets;
        const auto &past = input.past_sequence_lengths;
        const auto &total = input.total_sequence_lengths;
        const auto &positions = input.position_ids;
        const bool positions_i64 = cpu_vector(positions, DataType::kInt64);
        if (!ids || !*ids || (*ids)->ndim() != 2 || (*ids)->size(0) != 1
            || !cpu_vector(offsets, DataType::kInt32) || (*offsets)->numel() != 2
            || !cpu_vector(past, DataType::kInt32) || (*past)->numel() != 1
            || !cpu_vector(total, DataType::kInt32) || (*total)->numel() != 1
            || (!positions_i64 && !cpu_vector(positions, DataType::kInt32))
            || (*positions)->numel() != (*ids)->numel()) {
            throw std::invalid_argument("Classic NextN Graph prefill requires one packed span and CPU metadata");
        }
        const auto *bounds = reinterpret_cast<const int32_t *>((*offsets)->data());
        const auto past_rows = *reinterpret_cast<const int32_t *>((*past)->data());
        const auto total_rows = *reinterpret_cast<const int32_t *>((*total)->data());
        if (bounds[0] != 0 || bounds[1] <= (input.mtp_last_logits ? 1 : 0)
            || static_cast<size_t>(bounds[1]) != (*ids)->numel() || past_rows != 0
            || total_rows != bounds[1]) {
            throw std::invalid_argument("Classic NextN Graph prefill requires a nonempty zero-past span");
        }
        for (int32_t row = 0; row < bounds[1]; ++row) {
            const auto position = positions_i64
                ? reinterpret_cast<const int64_t *>((*positions)->data())[row]
                : reinterpret_cast<const int32_t *>((*positions)->data())[row];
            if (position != row) {
                throw std::invalid_argument("Classic NextN Graph prefill positions must begin at zero and be contiguous");
            }
        }
    }
    bool last_logits_expected = input.mtp_graph_draft_seed;
    if (input.mtp_last_logits) {
        using namespace infinilm::runtime;
        const auto type = model_config_->get<std::string>("model_type");
        const bool target = type == "deepseek_v4" && !rank_hidden;
        const bool draft = type == "deepseek_v4_mtp" && rank_hidden.has_value();
        if ((!target && !draft) || (enable_graph_compiling_ && !input.mtp_graph_fresh_prefill)
            || communication_group_.get_dist_config().pp_size != 1
            || input.target_hidden_states || !input.return_draft_hidden_states
            || input.rollback_v4_cache || input.mtp_scratch_step || input.mtp_draft_cache_only
            || input.sample_all_positions || input.top_k != 1 || input.sampling_rows
            || input.random_values) {
            throw std::invalid_argument(
                "Classic NextN last logits require eager greedy Target prefill or Draft repair with full returned HC");
        }
        const auto &ids = input.input_ids;
        const auto &offsets = input.input_offsets;
        const auto &past = input.past_sequence_lengths;
        const auto &positions = input.position_ids;
        const bool positions_i64 = cpu_vector(positions, DataType::kInt64);
        if (!ids || !*ids || (*ids)->ndim() != 2 || (*ids)->size(0) != 1
            || !cpu_vector(offsets, DataType::kInt32) || (*offsets)->numel() < 2
            || !cpu_vector(past, DataType::kInt32) || (*past)->numel() != (*offsets)->numel() - 1
            || (!positions_i64 && !cpu_vector(positions, DataType::kInt32))
            || (*positions)->numel() != (*ids)->numel()) {
            throw std::invalid_argument("Classic NextN last logits require packed IDs and matching CPU metadata");
        }
        const auto *bounds = reinterpret_cast<const int32_t *>((*offsets)->data());
        const auto *past_lengths = reinterpret_cast<const int32_t *>((*past)->data());
        const auto requests = (*past)->numel();
        const auto rows = (*ids)->numel();
        if (bounds[0] != 0 || bounds[requests] <= 0 || static_cast<size_t>(bounds[requests]) != rows) {
            throw std::invalid_argument("Classic NextN last logits offsets must cover every packed token");
        }
        for (size_t request = 0; request < requests; ++request) {
            const auto begin = bounds[request];
            const auto end = bounds[request + 1];
            if (end <= begin || static_cast<size_t>(end) > rows) {
                throw std::invalid_argument("Classic NextN last logits require nonempty packed spans");
            }
            const auto position = positions_i64
                ? reinterpret_cast<const int64_t *>((*positions)->data())[begin]
                : reinterpret_cast<const int32_t *>((*positions)->data())[begin];
            if (past_lengths[request] < 0 || position != past_lengths[request]
                || (target && past_lengths[request] != 0)) {
                throw std::invalid_argument("Classic NextN last logits require matching positions and fresh Target prefill");
            }
        }
        last_logits_expected = rows > requests;
    }
    if (input.mtp_draft_cache_only) {
        if (model_config_->get<std::string>("model_type") != "deepseek_v4_mtp" ||
            communication_group_.get_dist_config().pp_size != 1 ||
            (enable_graph_compiling_ && !input.mtp_graph_fresh_prefill) ||
            !rank_hidden || input.return_draft_hidden_states || input.rollback_v4_cache) {
            throw std::invalid_argument("Classic NextN cache-only requires an eager Draft with rank-local input HC and no outputs");
        }
        using namespace infinilm::runtime;
        const auto &offsets = input.input_offsets;
        if (!input.input_ids || !*input.input_ids || !offsets || !*offsets ||
            (*offsets)->device().type() != Device::Type::kCpu ||
            (*offsets)->dtype() != DataType::kInt32 || !(*offsets)->is_contiguous() ||
            (*offsets)->ndim() != 1 || (*offsets)->size(0) != 2) {
            throw std::invalid_argument("Classic NextN cache-only requires one span with CPU int32 offsets");
        }
        const auto *bounds = reinterpret_cast<const int32_t *>((*offsets)->data());
        if (bounds[0] != 0 || bounds[1] <= 0 ||
            static_cast<size_t>(bounds[1]) != (*input.input_ids)->numel()) {
            throw std::invalid_argument("Classic NextN cache-only offsets must cover all input tokens");
        }
    }
    if (input.mtp_scratch_step) {
        using namespace infinilm::runtime;
        const auto *paged = dynamic_cast<const cache::PagedKVCacheConfig *>(cache_config_.get());
        if (model_config_->get<std::string>("model_type") != "deepseek_v4"
            || communication_group_.get_dist_config().pp_size != 1 || !paged
            || paged->block_size() != models::deepseek_v4::kV4PageSize
            || input.rollback_v4_cache || input.mtp_draft_cache_only
            || input.target_hidden_states || rank_hidden || !input.sample_all_positions || input.top_k != 1) {
            throw std::invalid_argument("Classic NextN scratch steps require a greedy V4 Target without rollback");
        }
        const auto cpu_tensor = [](const std::optional<Tensor> &value, DataType type, size_t ndim) {
            return value && *value && (*value)->device().type() == Device::Type::kCpu
                && (*value)->dtype() == type && (*value)->is_contiguous() && (*value)->ndim() == ndim;
        };
        const auto &ids = input.input_ids;
        const auto &offsets = input.input_offsets;
        const auto &positions = input.position_ids;
        const auto &tables = input.block_tables;
        const auto &slots = input.slot_mapping;
        if (!cpu_tensor(ids, DataType::kInt64, 2) || (*ids)->size(0) != 1 ||
            !cpu_tensor(offsets, DataType::kInt32, 1) || !cpu_tensor(positions, DataType::kInt64, 1) ||
            !cpu_tensor(tables, DataType::kInt32, 2) || !cpu_tensor(slots, DataType::kInt64, 1)) {
            throw std::invalid_argument("Classic NextN scratch step requires contiguous CPU input metadata");
        }
        const auto rows = (*ids)->numel();
        const auto width = (*tables)->size(1);
        if (rows < 1 || rows > 3 || (*offsets)->numel() != rows + 1 ||
            (*positions)->numel() != rows || (*tables)->size(0) != rows || (*slots)->numel() != rows) {
            throw std::invalid_argument("Classic NextN scratch step requires 1-3 q1 siblings");
        }
        const auto *bounds = reinterpret_cast<const int32_t *>((*offsets)->data());
        const auto *pos = reinterpret_cast<const int64_t *>((*positions)->data());
        const auto *pages = reinterpret_cast<const int32_t *>((*tables)->data());
        const auto *slot = reinterpret_cast<const int64_t *>((*slots)->data());
        const auto page_size = paged->block_size();
        if (pos[0] <= 0 || pos[0] >= std::numeric_limits<int32_t>::max() ||
            static_cast<size_t>(pos[0]) / page_size >= width || bounds[0] != 0) {
            throw std::invalid_argument("Classic NextN scratch position exceeds its private table");
        }
        const auto page_index = static_cast<size_t>(pos[0]) / page_size;
        std::unordered_set<int32_t> private_pages;
        for (size_t row = 0; row < rows; ++row) {
            if (bounds[row + 1] != static_cast<int32_t>(row + 1) || pos[row] != pos[0]) {
                throw std::invalid_argument("Classic NextN scratch step must advance all active siblings by one row");
            }
            const auto *block_table = pages + row * width;
            bool padding = false;
            for (size_t column = 0; column < width; ++column) {
                const int32_t page = block_table[column];
                if (page == -1 && column > page_index) {
                    padding = true;
                    continue;
                }
                if (page < 0 || padding || static_cast<size_t>(page) >= paged->num_blocks() ||
                    !private_pages.insert(page).second) {
                    throw std::invalid_argument("Classic NextN scratch pages must be distinct and valid");
                }
            }
            if (slot[row] != static_cast<int64_t>(block_table[page_index]) * static_cast<int64_t>(page_size)
                                + pos[0] % static_cast<int64_t>(page_size)) {
                throw std::invalid_argument("Classic NextN scratch slot differs from its private page");
            }
        }
    }
    if (input.mtp_graph_draft_seed) {
        const auto *paged = dynamic_cast<const cache::PagedKVCacheConfig *>(cache_config_.get());
        const auto &dist = communication_group_.get_dist_config();
        const auto *enabled = std::getenv("INFINILM_MTP_GRAPH_K3_DRAFT_SEED");
        if (!enabled || std::string_view(enabled) != "1" || !enable_graph_compiling_
            || model_config_->get<std::string>("model_type") != "deepseek_v4_mtp"
            || nsa_pool_epoch_ || dist.pp_size != 1 || dist.tp_device_ids.size() != 8 || dist.moe_ep_size != 1
            || !paged || paged->max_batch_size() != 1 || paged->block_size() != models::deepseek_v4::kV4PageSize
            || !rank_hidden || input.target_hidden_states || !input.return_draft_hidden_states
            || input.sample_all_positions || input.top_k != 1 || input.sampling_rows || input.random_values
            || input.rollback_v4_cache || input.mtp_scratch_step || input.mtp_last_logits
            || input.mtp_graph_direct_commit || input.mtp_draft_cache_only || input.mtp_graph_fresh_prefill
            || input.mtp_cpu_metadata_shadow || input.mtp_shared_compression_plan) {
            throw std::invalid_argument("Classic NextN Graph Draft seed requires one TP8 greedy Draft owner and per-rank HC");
        }
        validate_packed_direct_metadata(input, *paged, model_config_->get<size_t>("vocab_size"), 1);
    }
    if (rank_hidden || input.return_draft_hidden_states || input.rollback_v4_cache) {
        const auto type = model_config_->get<std::string>("model_type");
        if ((type != "deepseek_v4" && type != "deepseek_v4_mtp")
            || communication_group_.get_dist_config().pp_size != 1
            || (rank_hidden && type != "deepseek_v4_mtp")) {
            throw std::invalid_argument("Classic V4 MTP input requires a single-stage V4 model");
        }
    }
    if (rank_hidden && (rank_hidden->size() != workers_.size() || input.target_hidden_states
                        || std::any_of(rank_hidden->begin(), rank_hidden->end(),
                                       [](const auto &tensor) { return !tensor; }))) {
        throw std::invalid_argument("MTP requires one non-null hidden-state tensor per rank");
    }
    if (rank_hidden) {
        if (!input.input_ids || !*input.input_ids) {
            throw std::invalid_argument("Classic V4 MTP requires input IDs");
        }
        const auto hidden_size = model_config_->get<size_t>("hidden_size");
        for (size_t i = 0; i < rank_hidden->size(); ++i) {
            const auto &tensor = (*rank_hidden)[i];
            if (tensor->device() != communication_group_.get_rank_info(i).device
                || tensor->dtype() != infinilm::runtime::DataType::kBFloat16
                || tensor->ndim() != 3 || tensor->size(0) != (*input.input_ids)->numel()
                || tensor->size(1) != 4 || tensor->size(2) != hidden_size) {
                throw std::invalid_argument("Invalid rank-local BF16 hidden states for Classic V4 MTP");
            }
        }
    }
    // All rank-local tensors are prepared before any worker can enter a collective.
    if (rank_hidden || rank_segments) {
        auto rank_input = input;
        rank_input.target_hidden_states_by_rank.reset();
        rank_input.target_hidden_segments_by_rank.reset();
        for (size_t i = 0; i < workers_.size(); ++i) {
            if (rank_hidden) {
                rank_input.target_hidden_states = (*rank_hidden)[i];
            } else {
                rank_input.target_hidden_segments = (*rank_segments)[i];
            }
            workers_[i]->run(rank_input, v4_host_batch);
        }
    } else {
        for (auto &worker : workers_) {
            worker->run(input, v4_host_batch);
        }
    }
    // Wait for all workers
    for (auto &worker : workers_) {
        worker->wait();
    }

    auto output = workers_[0]->get_output();
    output.mtp_cpu_metadata_shadow_used = bool(v4_host_batch);
    output.mtp_shared_compression_plan_used = v4_host_batch && bool(v4_host_batch->plans());
    // Completed worker output remains a mutable borrowed view, not a snapshot.
    const auto completed_view = [](infinilm::runtime::Tensor tensor) {
        if (!tensor) {
            return tensor;
        }
        return infinilm::runtime::Tensor::strided_from_blob(
            tensor->data(), tensor->shape(), tensor->strides(),
            tensor->dtype(), tensor->device(), tensor->shared_from_this());
    };
    output.output_ids = completed_view(output.output_ids);
    output.logits = completed_view(output.logits);
    const bool explicit_graph_hc = input.mtp_graph_direct_commit || input.mtp_graph_draft_seed;
    const bool owns_graph_hc = (input.mtp_scratch_step || explicit_graph_hc)
        && input.return_draft_hidden_states;
    if (!owns_graph_hc) {
        output.hidden_states = completed_view(output.hidden_states);
    }
    if (input.return_draft_hidden_states) {
        output.hidden_states_by_rank.reserve(workers_.size());
        if (input.mtp_graph_direct_commit) {
            output.mtp_graph_direct_rank_reports.reserve(workers_.size());
        }
        for (const auto &worker : workers_) {
            auto rank_output = worker->get_output();
            if (input.mtp_graph_draft_seed &&
                (!rank_output.graph_execution || rank_output.graph_execution->device_graph_segments == 0)) {
                throw std::runtime_error("Classic NextN Draft seed did not run a device Graph on every rank");
            }
            auto hidden = std::move(rank_output.hidden_states);
            if (!hidden) {
                throw std::runtime_error("V4 MTP did not return hidden states on every rank");
            }
            if ((input.mtp_last_logits || input.mtp_graph_draft_seed)
                && (rank_output.mtp_last_logits_used != last_logits_expected
                    || hidden->ndim() != 3 || hidden->size(0) != (*input.input_ids)->numel())) {
                throw std::runtime_error("Classic NextN last logits receipt or HC rows differ across ranks");
            }
            if (input.mtp_graph_direct_commit) {
                output.mtp_graph_direct_rank_reports.push_back(std::move(rank_output.graph_execution).value());
            }
            output.hidden_states_by_rank.push_back(
                owns_graph_hc ? std::move(hidden) : completed_view(std::move(hidden)));
        }
    }
    return output;
}

void InferEngine::compile() {
    std::lock_guard<std::mutex> submission_lock(submission_mutex_);
    if (nsa_pool_epoch_) {
        throw std::invalid_argument("NSA owner slice excludes COMPILE before state mutation");
    }
    if (weight_state_ == WeightState::Loading) {
        return;
    }
    if (weight_state_ != WeightState::Ready) {
        throw std::logic_error("cannot compile an incomplete or failed weight load");
    }
    try {
        compile_workers();
    } catch (...) {
        weight_state_ = WeightState::Failed;
        throw;
    }
}

void InferEngine::compile_workers() {
    if (nsa_pool_epoch_) {
        return; // weight finalization/reset must not enter COMPILE
    }
    for (auto &worker : workers_) {
        worker->compile();
    }
    // Wait for all workers
    for (auto &worker : workers_) {
        worker->wait();
    }
}

//------------------------------------------------------
// Destructor
//------------------------------------------------------
InferEngine::~InferEngine() {
    // Close all workers
    for (auto &worker : workers_) {
        worker->request_stop();
    }
    if (barrier_ != nullptr) {
        barrier_->abort(nullptr);
    }
    for (auto &worker : workers_) {
        worker->join();
    }
    // Workers retain models and recorded calls that can refer to communicators.
    // Destroy them while communication_group_ is still alive; member destruction
    // alone would run in the opposite order of these declarations.
    workers_.clear();
}

const distributed::DistConfig &InferEngine::get_dist_config() const {
    return communication_group_.get_dist_config();
}

std::unique_ptr<cache::CacheConfig> InferEngine::get_cache_config() const {
    std::lock_guard<std::mutex> submission_lock(submission_mutex_);
    return cache_config_ ? cache_config_->unique_copy() : nullptr;
}

//------------------------------------------------------
// reset_cache (overloaded with CacheConfig)
//------------------------------------------------------
void InferEngine::reset_cache(const cache::CacheConfig *new_config) {
    std::lock_guard<std::mutex> submission_lock(submission_mutex_);
    if (new_config == nullptr) {
        throw std::invalid_argument("reset_cache requires a cache configuration");
    }
    if (weight_state_ != WeightState::Loading
        && weight_state_ != WeightState::Ready) {
        throw std::logic_error("cannot reset cache during an incomplete or failed weight load");
    }
    if (nsa_pool_epoch_) {
        const auto *paged = dynamic_cast<const cache::PagedKVCacheConfig *>(new_config);
        if (!paged || !paged->block_size() || paged->block_size() % 64 || !paged->num_blocks()
            || paged->block_size() > 2147483647u || paged->num_blocks() > 2147483647u
            || (attention_backend_ == backends::AttentionBackend::FLASH_ATTN && paged->block_size() % 256)) {
            throw std::invalid_argument("NSA reset requires supported paged geometry before owner mutation");
        }
    }
    const auto replacement_epoch = nsa_pool_epoch_ ? issue_nsa_pool_epoch() : 0;
    try {
        for (auto &worker : workers_) {
            worker->reset_cache(new_config, replacement_epoch);
        }
        for (auto &worker : workers_) {
            worker->wait();
        }
        cache_config_ = new_config->unique_copy();
        nsa_pool_epoch_ = replacement_epoch; // publish only after all workers succeeded
        if (weight_state_ == WeightState::Loading) {
            return;
        }
        compile_workers();
    } catch (...) {
        weight_state_ = WeightState::Failed;
        throw;
    }
}

std::pair<size_t, size_t> InferEngine::deepseek_v4_raw_ring_rows() const {
    return {models::deepseek_v4::compressor_ring_rows(4, classic_mtp_target_),
            models::deepseek_v4::compressor_ring_rows(128, classic_mtp_target_)};
}

void InferEngine::clone_v4_cache_pages(const std::vector<std::pair<size_t, size_t>> &copies) {
    std::lock_guard<std::mutex> submission_lock(submission_mutex_);
    if (weight_state_ != WeightState::Ready &&
        (weight_state_ != WeightState::Loading || weight_load_started_)) {
        throw std::logic_error("cannot clone cache pages of an incomplete or failed engine");
    }
    const auto *paged = dynamic_cast<const cache::PagedKVCacheConfig *>(cache_config_.get());
    if (model_config_->get<std::string>("model_type") != "deepseek_v4" ||
        communication_group_.get_dist_config().pp_size != 1 || !paged ||
        paged->block_size() != models::deepseek_v4::kV4PageSize || workers_.empty()) {
        throw std::invalid_argument("V4 cache page clone requires a single-stage paged DeepSeek V4 Target");
    }
    models::deepseek_v4::validate_cache_page_copies(copies, paged->num_blocks());
    barrier_->rethrow_if_broken();
    const auto frozen = std::make_shared<const std::vector<std::pair<size_t, size_t>>>(copies);
    size_t submitted = 0;
    std::exception_ptr error;
    try {
        for (; submitted < workers_.size(); ++submitted) {
            workers_[submitted]->clone_v4_cache_pages(frozen);
        }
    } catch (...) {
        error = std::current_exception();
        // A submitted worker may be waiting for a rank that never received
        // the command. Abort the rendezvous before waiting for those workers.
        barrier_->abort(error);
    }
    for (size_t rank = 0; rank < submitted; ++rank) {
        try {
            workers_[rank]->wait();
        } catch (...) {
            if (!error) {
                error = std::current_exception();
            }
        }
    }
    if (error) {
        weight_state_ = WeightState::Failed;
        std::rethrow_exception(error);
    }
}

std::uint64_t InferEngine::get_nsa_pool_epoch() {
    std::lock_guard<std::mutex> submission_lock(submission_mutex_);
    if (!nsa_pool_epoch_ || workers_.empty() || weight_state_ == WeightState::Failed) {
        throw std::logic_error("NSA pool query requires a live VideoNSA engine");
    }
    for (const auto &worker : workers_) {
        if (worker->get_nsa_pool_epoch() != nsa_pool_epoch_) {
            throw std::logic_error("NSA engine/worker pool epoch mismatch");
        }
    }
    return nsa_pool_epoch_;
}

std::vector<std::vector<infinilm::runtime::Tensor>> InferEngine::get_kv_cache() {
    std::lock_guard<std::mutex> submission_lock(submission_mutex_);
    if (nsa_pool_epoch_) {
        throw std::invalid_argument("NSA borrowed KV writers are outside owner slice");
    }
    std::vector<std::vector<infinilm::runtime::Tensor>> kv_cache_list;
    if (workers_.empty()) {
        throw std::runtime_error("InferEngine::get_cache_vec: no workers");
    }

    kv_cache_list.reserve(workers_.size());
    for (auto &worker : workers_) {
        kv_cache_list.push_back(std::move(worker->get_kv_cache()));
    }

    for (auto &worker : workers_) {
        worker->wait();
    }

    return kv_cache_list;
}

} // namespace infinilm::engine
