#include "native_ops.hpp"

#include "execution/submit.hpp"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <string_view>

namespace infinilm::models::deepseek_v4 {

void apply_rotary_tail(runtime::Runtime &runtime, const runtime::Tensor &input,
                       const runtime::Tensor &positions, const runtime::Tensor &cos_sin, bool inverse) {
    using runtime::DataType;
    if (!input || (input->ndim() != 2 && input->ndim() != 3) || !input->is_contiguous()
        || (input->dtype() != DataType::kFloat32 && input->dtype() != DataType::kBFloat16)
        || !positions || positions->shape() != runtime::Shape{input->size(0)}
        || !positions->is_contiguous()
        || (positions->dtype() != DataType::kInt32 && positions->dtype() != DataType::kInt64)
        || !cos_sin || cos_sin->ndim() != 2 || !cos_sin->size(0) || cos_sin->size(1) != 64
        || cos_sin->dtype() != DataType::kFloat32 || !cos_sin->is_contiguous()) {
        throw std::invalid_argument("Invalid DeepSeek V4 tail rotary inputs");
    }
    const auto dim = static_cast<int64_t>(input->size(input->ndim() - 1));
    if (dim != 128 && dim != 512) {
        throw std::invalid_argument("DeepSeek V4 rotary input head must be 128 or 512 wide");
    }
    auto ids = positions;
    if (positions->dtype() == DataType::kInt32) {
        ids = runtime::Tensor::empty(runtime, positions->shape(), DataType::kInt64);
        using Cast = infini::ops::Cast;
        execution::submit<Cast>(runtime, execution::RecordingPolicy::DeviceOnly,
                                Cast::DefaultConfig(runtime.device().type()), positions, ids);
    }
    using Op = infini::ops::RotaryEmbedding;
    execution::submit<Op>(runtime, execution::RecordingPolicy::DeviceOnly,
                          Op::DefaultConfig(runtime.device().type()), ids, input,
                          std::optional<runtime::Tensor>{}, cos_sin, dim, false, dim - 64, inverse);
}

void store_swa_rows(runtime::Runtime &runtime, const runtime::Tensor &key,
                    const runtime::Tensor &weight, const runtime::Tensor &positions,
                    const runtime::Tensor &frequencies, const runtime::Tensor &slots,
                    float epsilon, const runtime::Tensor &pages) {
    using Op = infini::ops::DeepseekV4StoreSwa;
    execution::submit<Op>(runtime, execution::RecordingPolicy::DeviceOnly,
                          Op::DefaultConfig(runtime.device().type()),
                          key, weight, positions, frequencies, slots, epsilon, pages);
}

void store_mla_rows(runtime::Runtime &runtime, const runtime::Tensor &kv,
                    const runtime::Tensor &slots, const runtime::Tensor &pages,
                    size_t page_tokens) {
    if (page_tokens != 256 && page_tokens != 64 && page_tokens != 2) {
        throw std::invalid_argument("Unsupported DeepSeek V4 packed cache page size");
    }
    using Op = infini::ops::DeepseekV4StoreMla;
    execution::submit<Op>(runtime, execution::RecordingPolicy::DeviceOnly,
                          Op::DefaultConfig(runtime.device().type()), kv, slots,
                          static_cast<int64_t>(page_tokens), pages);
}

runtime::Tensor prepare_sparse_mla_schedule(runtime::Runtime &runtime,
                                            const runtime::Tensor &swa_lengths, size_t swa_width,
                                            const runtime::Tensor &extra_lengths, size_t extra_width) {
    using Op = infini::ops::DeepseekV4PrepareSparseMlaSchedule;
    using Opt = std::optional<runtime::Tensor>;
    auto schedule = runtime::Tensor::empty(runtime, {Op::kScheduleBytes}, runtime::DataType::kUInt8);
    execution::submit<Op>(runtime, execution::RecordingPolicy::DeviceOnly,
                          Op::DefaultConfig(runtime.device().type()), swa_lengths,
                          extra_lengths ? Opt{extra_lengths} : Opt{}, static_cast<int64_t>(swa_width),
                          static_cast<int64_t>(extra_width), schedule);
    return schedule;
}

runtime::Tensor sparse_mla(runtime::Runtime &runtime, const runtime::Tensor &query,
                           const LayerPagedCache &cache, const runtime::Tensor &swa_indices,
                           const runtime::Tensor &swa_lengths, const runtime::Tensor &sink,
                           const runtime::Tensor &compressed_indices,
                           const runtime::Tensor &compressed_lengths, double scale,
                           const runtime::Tensor &schedule,
                           size_t first_prefill_page, size_t prefill_pages) {
    const auto ratio = cache.compression_ratio;
    if (!query || (ratio != 0 && ratio != 4 && ratio != 128)
        || static_cast<bool>(ratio) != static_cast<bool>(cache.compressed)
        || static_cast<bool>(ratio) != static_cast<bool>(compressed_indices)
        || static_cast<bool>(ratio) != static_cast<bool>(compressed_lengths)) {
        throw std::invalid_argument("DeepSeek V4 compressed cache and sparse metadata disagree");
    }
    using Opt = std::optional<runtime::Tensor>;
    auto output = runtime::Tensor::empty(runtime, query->shape(), query->dtype());
    const auto extra = ratio ? Opt{cache.compressed} : Opt{};
    const auto extra_indices = ratio ? Opt{compressed_indices} : Opt{};
    const auto extra_lengths = ratio ? Opt{compressed_lengths} : Opt{};
    const auto extra_page_size = static_cast<int64_t>(ratio ? kV4PageSize / ratio : 0);
    if (schedule) {
        using Op = infini::ops::DeepseekV4SparseMlaWithSchedule;
        execution::submit<Op>(runtime, execution::RecordingPolicy::DeviceOnly,
                              Op::DefaultConfig(runtime.device().type()), query, cache.swa,
                              swa_indices, swa_lengths, sink, extra, extra_indices, extra_lengths,
                              extra_page_size, scale, schedule, output);
    } else {
        using Op = infini::ops::DeepseekV4SparseMla;
        const auto default_config = Op::DefaultConfig(runtime.device().type());
        infini::ops::DeepseekV4PrefillPageWindowConfig window(first_prefill_page,prefill_pages);
        window.set_implementation_index(default_config.implementation_index());
        const auto &config = prefill_pages && query->size(0) > 3072
            ? static_cast<const infini::ops::Config &>(window) : default_config;
        execution::submit<Op>(runtime, execution::RecordingPolicy::DeviceOnly,
                              config, query, cache.swa,
                              swa_indices, swa_lengths, sink, extra, extra_indices, extra_lengths,
                              extra_page_size, scale, output);
    }
    return output;
}

runtime::Tensor run_routed_experts(runtime::Runtime &runtime, const runtime::Tensor &input,
                                   const RoutedExperts &routing, const W8A8ExpertWeights &weights,
                                   double swiglu_limit, std::optional<double> routed_scale) {
    if (!input) throw std::invalid_argument("DeepSeek V4 routed experts require input");
    using Op = infini::ops::DeepseekV4W8a8Experts;
    auto output = runtime::Tensor::empty(runtime, input->shape(), input->dtype());
    const auto submit_rows = [&](const runtime::Tensor &rows, const RoutedExperts &routes,
                                 const runtime::Tensor &result) {
        execution::submit<Op>(runtime, execution::RecordingPolicy::DeviceOnly,
                              Op::DefaultConfig(runtime.device().type()), rows, routes.weights,
                              routes.indices, weights.gate_up(), weights.gate_up_scale(),
                              weights.down(), weights.down_scale(), swiglu_limit, routed_scale, result);
    };
    // Bound the Hygon expert workspace and retain the grouped matrix kernel
    // across a large packed prefill. Each route is local to its token.
    constexpr size_t grouped_max_rows = 8192;
    const auto rows = input->size(0);
    const bool grouped_shape = runtime.device().type() == runtime::Device::Type::kHygon
        && input->dtype() == runtime::DataType::kBFloat16 && input->size(1) == 4096
        && weights.down()->size(2) == 256
        && (routed_scale ? weights.down()->size(0) == 256 && routing.weights->size(1) == 6
                         : weights.down()->size(0) == 1 && routing.weights->size(1) == 1);
    if (rows > grouped_max_rows && grouped_shape) {
        const auto *setting = std::getenv("INFINILM_DSV4_EXPERT_PREFILL_TILING");
        if (!setting || std::string_view(setting) != "0") {
            const size_t tile_count = 1 + (rows - 1) / grouped_max_rows;
            const size_t tile_rows = 1 + (rows - 1) / tile_count;
            for (size_t first = 0; first < rows; first += tile_rows) {
                const size_t count = std::min(tile_rows, rows - first);
                const auto take = [&](const runtime::Tensor &tensor) {
                    return tensor->narrow({{0, first, count}});
                };
                submit_rows(take(input), {take(routing.weights), take(routing.indices)}, take(output));
            }
            return output;
        }
    }
    submit_rows(input, routing, output);
    return output;
}

} // namespace infinilm::models::deepseek_v4
