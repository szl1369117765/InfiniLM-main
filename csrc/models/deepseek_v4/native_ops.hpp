#pragma once

#include "expert_weights.hpp"
#include "paged_cache.hpp"
#include "router.hpp"

namespace infinilm::models::deepseek_v4 {

// Rotate the last 64 channels of contiguous [rows, head_dim] or
// [rows, heads, head_dim] using a split cosine/sine table. Positions must be in range.
void apply_rotary_tail(runtime::Runtime &runtime, const runtime::Tensor &input,
                       const runtime::Tensor &positions, const runtime::Tensor &cos_sin,
                       bool inverse = false);

void store_swa_rows(runtime::Runtime &runtime, const runtime::Tensor &key,
                    const runtime::Tensor &weight, const runtime::Tensor &positions,
                    const runtime::Tensor &frequencies, const runtime::Tensor &slots,
                    float epsilon, const runtime::Tensor &pages);

void store_mla_rows(runtime::Runtime &runtime, const runtime::Tensor &kv,
                    const runtime::Tensor &slots, const runtime::Tensor &pages,
                    size_t page_tokens);

runtime::Tensor prepare_sparse_mla_schedule(runtime::Runtime &runtime,
                                            const runtime::Tensor &swa_lengths, size_t swa_width,
                                            const runtime::Tensor &extra_lengths, size_t extra_width);

runtime::Tensor sparse_mla(runtime::Runtime &runtime, const runtime::Tensor &query,
                           const LayerPagedCache &cache, const runtime::Tensor &swa_indices,
                           const runtime::Tensor &swa_lengths, const runtime::Tensor &sink,
                           const runtime::Tensor &compressed_indices,
                           const runtime::Tensor &compressed_lengths, double scale,
                           const runtime::Tensor &schedule,
                           size_t first_prefill_page = 0, size_t prefill_pages = 0);

// A rank-local expert contribution. The decoder must reduce across TP ranks
// and combine shared expert output before applying the residual.
runtime::Tensor run_routed_experts(runtime::Runtime &runtime, const runtime::Tensor &input,
                                   const RoutedExperts &routing, const W8A8ExpertWeights &weights,
                                   double swiglu_limit, std::optional<double> routed_scale);

} // namespace infinilm::models::deepseek_v4
