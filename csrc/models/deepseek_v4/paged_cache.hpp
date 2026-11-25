#pragma once

#include "cache_geometry.hpp"
#include "runtime/tensor.hpp"

#include <utility>
#include <vector>

namespace infinilm::engine {
class RankBarrier;
}

namespace infinilm::models::deepseek_v4 {

constexpr size_t kV4FlashMlaWidth = 584;
constexpr size_t kV4HeadDim = 512;

struct LayerPagedCache {
    runtime::Tensor swa;
    runtime::Tensor compressed;
    runtime::Tensor indexer;
    runtime::Tensor compressor_state;
    runtime::Tensor indexer_state;
    size_t compression_ratio{};
    size_t swa_blocks{};
    size_t full_blocks{};

    // FlashMLA expects a 4D byte view, with an aligned stride between pages.
    runtime::Tensor swa_flashmla_view() const;
    runtime::Tensor compressed_flashmla_view() const;
};

// Allocate storage before graph capture. Scheduler-owned full-to-SWA page
// mapping and compressor state updates are separate from this allocation.
LayerPagedCache allocate_layer_paged_cache(
    runtime::Runtime &runtime, size_t full_blocks, size_t swa_blocks,
    size_t compression_ratio, size_t index_head_dim = 128);

// The caller owns the source pages and all private destination leases.
void validate_cache_page_copies(
    const std::vector<std::pair<size_t, size_t>> &copies, size_t page_count);
void clone_cache_pages(
    runtime::Runtime &runtime, const std::vector<LayerPagedCache> &layers, size_t page_count,
    const std::vector<std::pair<size_t, size_t>> &copies, engine::RankBarrier &barrier);

// Writable locations from the host attention metadata and compression plans.
// All tensors are contiguous int32 row vectors. A value of -1 skips that write.
struct PagedCacheWriteSet {
    runtime::Tensor swa_slots;
    runtime::Tensor c4_slots;
    runtime::Tensor c128_slots;
    runtime::Tensor c4_state_slots;
    runtime::Tensor c128_state_slots;
};

// An eager verification submission on one rank and its owner stream.
// The caller must keep the Runtime and live cache layout in place until this
// finishes. Other submissions and cache reconfiguration must wait.
class PagedCacheSnapshot {
public:
    PagedCacheSnapshot(runtime::Runtime &runtime,
                       const std::vector<LayerPagedCache> &layers,
                       const PagedCacheWriteSet &writes,
                       runtime::Tensor *reusable_backup = nullptr);
    ~PagedCacheSnapshot() noexcept;

    PagedCacheSnapshot(const PagedCacheSnapshot &) = delete;
    PagedCacheSnapshot &operator=(const PagedCacheSnapshot &) = delete;

    void commit();
    void rollback();

private:
    struct Slice {
        runtime::Tensor live;
        size_t live_offset;
        size_t backup_offset;
        size_t bytes;
    };

    void require_active() const;
    void release();

    runtime::Runtime &runtime_;
    runtime::Tensor backup_;
    std::vector<Slice> slices_;
    bool active_{false};
};

} // namespace infinilm::models::deepseek_v4
