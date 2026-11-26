#include "compressor.hpp"
#include "native_ops.hpp"
#include "request_rows.hpp"

#include "execution/submit.hpp"
#include "layers/linear/dense.hpp"

#include <stdexcept>

namespace infinilm::models::deepseek_v4 {

Compressor::Compressor(runtime::Runtime &runtime, size_t hidden, size_t head_dim, size_t ratio,
                       double eps)
    : head_dim_(head_dim) {
    if (!hidden || (head_dim != 128 && head_dim != 512) ||
        (ratio != 4 && (ratio != 128 || head_dim != 512))) {
        throw std::invalid_argument("Invalid DeepSeek V4 compressor geometry");
    }
    const auto width = ratio == 4 ? 2 * head_dim : head_dim;
    const auto dtype = runtime::DataType::kBFloat16;
    ape_ = bind_weight("ape", weights::Weight(runtime, {ratio, width}, runtime::DataType::kFloat32));
    projection_ = std::make_shared<layers::linear::GateUpParallelLinear>(
        runtime, hidden, width, "wkv", "wgate",
        [this](const std::string &name, weights::Weight weight) {
            bind_weight(name, std::move(weight));
        },
        nullptr, false, dtype, engine::distributed::RankInfo(runtime.device()));
    norm_ = bind_child("norm", std::make_shared<layers::RMSNorm>(
        runtime, head_dim, eps, runtime::DataType::kFloat32));
}

std::vector<std::shared_ptr<weights::ModelWeights>> Compressor::loading_children() const {
    auto children = weights::ModelWeights::loading_children();
    children.push_back(projection_);
    return children;
}

runtime::Tensor Compressor::project_and_pool(runtime::Runtime &runtime,
                                             const runtime::Tensor &hidden_states,
                                             const runtime::Tensor &sources,
                                             const runtime::Tensor &state_slots,
                                             const runtime::Tensor &state,
                                             const std::vector<size_t> &query_offsets) const {
    if (!hidden_states || hidden_states->ndim() != 2 || !sources || sources->ndim() != 2) {
        throw std::invalid_argument("DeepSeek V4 compressor requires rank-two input and source rows");
    }
    auto projection = project_requests(runtime, query_offsets, [&](const RequestRows &part) {
        return layers::linear::dense_projection(
            runtime, part.slice(hidden_states), projection_->weight(), std::nullopt, 1.0f,
            runtime::DataType::kFloat32);
    });
    auto output = runtime::Tensor::empty(runtime, {sources->size(0), head_dim_}, projection->dtype());
    using Op = infini::ops::DeepseekV4Compress;
    execution::submit<Op>(runtime, execution::RecordingPolicy::DeviceOnly,
                          Op::DefaultConfig(runtime.device().type()), projection, ape_, sources,
                          state_slots, state, output);
    return output;
}

void Compressor::update_state(runtime::Runtime &runtime, const runtime::Tensor &hidden_states,
                              const runtime::Tensor &state_slots, const runtime::Tensor &state,
                              const std::vector<size_t> &query_offsets) const {
    auto sources = runtime::Tensor::empty(runtime, {0, 128}, runtime::DataType::kInt32);
    project_and_pool(runtime, hidden_states, sources, state_slots, state, query_offsets);
}

runtime::Tensor Compressor::forward(runtime::Runtime &runtime, const runtime::Tensor &hidden_states,
                                    const runtime::Tensor &sources, const runtime::Tensor &state_slots,
                                    const runtime::Tensor &state, const runtime::Tensor &positions,
                                    const runtime::Tensor &cos_sin,
                                    const std::vector<size_t> &query_offsets) const {
    auto output = project_and_pool(runtime, hidden_states, sources, state_slots, state, query_offsets);
    auto kv = norm_->forward(runtime, output);
    apply_rotary_tail(runtime, kv, positions, cos_sin);
    return kv;
}

void Compressor::store_paged(runtime::Runtime &runtime, const runtime::Tensor &hidden_states,
                             const runtime::Tensor &sources, const runtime::Tensor &state_slots,
                             const runtime::Tensor &state, const runtime::Tensor &positions,
                             const runtime::Tensor &cos_sin, const std::vector<size_t> &query_offsets,
                             const runtime::Tensor &slots, const runtime::Tensor &pages,
                             size_t page_tokens) const {
    if (runtime.device().type() == runtime::Device::Type::kHygon) {
        using Op = infini::ops::DeepseekV4StoreCompressed;
        auto pooled = project_and_pool(runtime, hidden_states, sources, state_slots, state, query_offsets);
        execution::submit<Op>(runtime, execution::RecordingPolicy::DeviceOnly,
                              Op::DefaultConfig(runtime.device().type()), pooled, norm_->weight(),
                              positions, cos_sin, slots, static_cast<float>(norm_->eps()),
                              static_cast<int64_t>(page_tokens), pages);
        return;
    }
    auto kv = forward(runtime, hidden_states, sources, state_slots, state, positions, cos_sin, query_offsets);
    auto rounded = runtime::Tensor::empty(runtime, kv->shape(), runtime::DataType::kBFloat16);
    using Cast = infini::ops::Cast;
    execution::submit<Cast>(runtime, execution::RecordingPolicy::DeviceOnly,
                            Cast::DefaultConfig(runtime.device().type()), kv, rounded);
    store_mla_rows(runtime, rounded, slots, pages, page_tokens);
}

} // namespace infinilm::models::deepseek_v4
