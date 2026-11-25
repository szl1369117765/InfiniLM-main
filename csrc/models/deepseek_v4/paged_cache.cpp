#include "paged_cache.hpp"

#include "engine/rank_barrier.hpp"
#include "runtime/runtime.hpp"

#include <algorithm>
#include <cstdint>
#include <exception>
#include <limits>
#include <stdexcept>
#include <unordered_set>

namespace infinilm::models::deepseek_v4 {
namespace {

constexpr size_t kFlashMlaAlignment = 576;
constexpr size_t kIndexQuantBlock = 128;

size_t product(size_t x, size_t y) {
    if (y && x > std::numeric_limits<size_t>::max() / y) {
        throw std::overflow_error("DeepSeek V4 cache size overflows");
    }
    return x * y;
}

size_t align(size_t x, size_t multiple) {
    if (x > std::numeric_limits<size_t>::max() - (multiple - 1)) {
        throw std::overflow_error("DeepSeek V4 cache alignment overflows");
    }
    return (x + multiple - 1) / multiple * multiple;
}

runtime::Tensor zeros(runtime::Runtime &rt, size_t rows, size_t width, runtime::DataType type) {
    const auto bytes = product(product(rows, width), runtime::dsize(type));
    auto result = runtime::Tensor::empty(rt, {rows, width}, type);
    rt.setDeviceMemory(result->data(), 0, bytes);
    return result;
}

runtime::Tensor view_as_flashmla(const runtime::Tensor &storage, size_t page_tokens) {
    if (!storage) return {};
    const auto width = static_cast<runtime::Stride>(kV4FlashMlaWidth);
    const auto stride = static_cast<runtime::Stride>(storage->size(1));
    return storage->as_strided({storage->size(0), page_tokens, 1, kV4FlashMlaWidth},
                               {stride, width, width, 1});
}

std::vector<size_t> written_rows(const runtime::Tensor &locations, size_t query_rows,
                                 size_t slots_per_row) {
    const runtime::Device cpu{runtime::Device::Type::kCpu};
    if (!locations || locations->device() != cpu ||
        locations->dtype() != runtime::DataType::kInt32 ||
        !locations->is_contiguous() || locations->shape() != runtime::Shape{query_rows}) {
        throw std::invalid_argument("DeepSeek V4 snapshot requires host int32 write locations");
    }
    const auto *slots = reinterpret_cast<const int32_t *>(locations->data());
    std::vector<size_t> rows;
    rows.reserve(query_rows);
    for (size_t index = 0; index < query_rows; ++index) {
        if (slots[index] < -1) {
            throw std::invalid_argument("Invalid DeepSeek V4 snapshot write location");
        }
        if (slots[index] >= 0) {
            rows.push_back(static_cast<size_t>(slots[index]) / slots_per_row);
        }
    }
    std::sort(rows.begin(), rows.end());
    rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
    return rows;
}

void drain_or_terminate(runtime::Runtime &rt) noexcept {
    try {
        rt.syncStream();
    } catch (...) {
        std::terminate();
    }
}

} // namespace

runtime::Tensor LayerPagedCache::swa_flashmla_view() const {
    return view_as_flashmla(swa, kV4PageSize);
}

runtime::Tensor LayerPagedCache::compressed_flashmla_view() const {
    if (!compression_ratio) return {};
    return view_as_flashmla(compressed, kV4PageSize / compression_ratio);
}

LayerPagedCache allocate_layer_paged_cache(runtime::Runtime &rt, size_t full_blocks,
                                           size_t swa_blocks, size_t ratio, size_t index_head_dim,
                                           bool classic_mtp_target) {
    rt.assertOwnerThread();
    if (rt.isGraphRecording()) {
        throw std::logic_error("DeepSeek V4 cache allocation cannot run during graph recording");
    }
    if (!full_blocks || !swa_blocks || swa_blocks > full_blocks
        || (ratio != 0 && ratio != 4 && ratio != 128)
        || (ratio == 4 && (!index_head_dim || index_head_dim % kIndexQuantBlock))) {
        throw std::invalid_argument("Invalid DeepSeek V4 cache geometry");
    }

    using runtime::DataType;
    LayerPagedCache cache;
    cache.compression_ratio = ratio;
    cache.swa_blocks = swa_blocks;
    cache.full_blocks = full_blocks;
    cache.swa = zeros(rt, swa_blocks, align(kV4PageSize * kV4FlashMlaWidth, kFlashMlaAlignment),
                      DataType::kUInt8);
    if (!ratio) return cache;

    const auto page_tokens = kV4PageSize / ratio;
    cache.compressed = zeros(rt, full_blocks, align(page_tokens * kV4FlashMlaWidth, kFlashMlaAlignment),
                             DataType::kUInt8);
    const size_t ring = compressor_ring_rows(ratio, classic_mtp_target);
    cache.compressor_ring_size = ring;
    const size_t coeff = ratio == 4 ? 2 : 1;
    if (swa_blocks > std::numeric_limits<size_t>::max() / ring - 2) {
        throw std::overflow_error("DeepSeek V4 compressor ring size overflows");
    }
    const auto state_rows = align((swa_blocks + 1) * ring + 1, ratio);
    cache.compressor_state = zeros(rt, state_rows, 2 * coeff * kV4HeadDim, DataType::kFloat32);
    if (ratio == 4) {
        const auto index_page_bytes =
            product(page_tokens, product(index_head_dim / kIndexQuantBlock, kIndexQuantBlock + sizeof(float)));
        cache.indexer = zeros(rt, full_blocks, index_page_bytes, DataType::kUInt8);
        cache.indexer_state = zeros(rt, state_rows, product(2 * coeff, index_head_dim), DataType::kFloat32);
    }
    return cache;
}

void validate_cache_page_copies(
    const std::vector<std::pair<size_t, size_t>> &copies, size_t page_count) {
    if (copies.empty() || copies.size() > page_count) {
        throw std::invalid_argument("V4 cache page clone requires a bounded nonempty copy list");
    }
    std::unordered_set<size_t> sources;
    std::unordered_set<size_t> destinations;
    for (const auto &[source, destination] : copies) {
        if (source >= page_count || destination >= page_count) {
            throw std::out_of_range("V4 cache page clone exceeds the current page pool");
        }
        sources.insert(source);
        if (!destinations.insert(destination).second) {
            throw std::invalid_argument("V4 cache page clone destinations must be unique");
        }
    }
    for (const auto destination : destinations) {
        if (sources.count(destination)) {
            throw std::invalid_argument("V4 cache page clone sources and destinations overlap");
        }
    }
}

void clone_cache_pages(
    runtime::Runtime &rt, const std::vector<LayerPagedCache> &layers, size_t page_count,
    const std::vector<std::pair<size_t, size_t>> &copies, engine::RankBarrier &barrier) {
    rt.assertOwnerThread();
    if (rt.isGraphRecording() || layers.empty()) {
        throw std::logic_error("V4 cache page clone needs initialized eager paged caches");
    }
    // InferEngine has already validated the copy-list structure. Recheck the
    // bounds against each worker's actual pool and inspect its local stores
    // before permitting any rank to submit a write.
    for (const auto &[source, destination] : copies) {
        if (source >= page_count || destination >= page_count) {
            throw std::out_of_range("V4 cache page clone exceeds a worker's page pool");
        }
    }
    struct Store {
        std::byte *data;
        size_t page_bytes;
    };
    std::vector<Store> stores;
    std::vector<std::pair<std::uintptr_t, std::uintptr_t>> ranges;
    using runtime::DataType;
    auto append = [&](runtime::Tensor storage, DataType type, size_t rows_per_page, size_t padded_rows) {
        if (!storage || storage->device() != rt.device() || storage->dtype() != type ||
            storage->ndim() != 2 || !storage->is_contiguous() ||
            storage->size(0) != padded_rows || !storage->size(1) ||
            storage->allocation_owner().get() != &rt || !storage->data()) {
            throw std::invalid_argument("Invalid V4 cache page clone storage");
        }
        auto *data = storage->data();
        const auto address = reinterpret_cast<std::uintptr_t>(data);
        const auto row_bytes = product(storage->size(1), runtime::dsize(type));
        const auto bytes = product(storage->size(0), row_bytes);
        if (bytes > std::numeric_limits<std::uintptr_t>::max() - address) {
            throw std::overflow_error("V4 cache page clone storage address overflows");
        }
        ranges.emplace_back(address, address + bytes);
        stores.push_back({data, product(rows_per_page, row_bytes)});
    };
    for (const auto &layer : layers) {
        if (layer.full_blocks != page_count || layer.swa_blocks != page_count) {
            throw std::invalid_argument("V4 cache page clone requires paired full and SWA pages");
        }
        append(layer.swa, DataType::kUInt8, 1, page_count);
        const auto ratio = layer.compression_ratio;
        if (ratio == 0) {
            continue;
        }
        if (ratio != 4 && ratio != 128) {
            throw std::invalid_argument("Invalid V4 cache page clone compression ratio");
        }
        const auto ring = layer.compressor_ring_size;
        if (ring != compressor_ring_rows(ratio, false) && ring != compressor_ring_rows(ratio, true)) {
            throw std::invalid_argument("Invalid V4 cache page clone compressor ring");
        }
        if (page_count > std::numeric_limits<size_t>::max() / ring - 2) {
            throw std::overflow_error("V4 cache page clone ring size overflows");
        }
        // Match allocation, including the extra raw-ring padding rows. A
        // copied physical page never spans the padding.
        const auto state_rows = align((page_count + 1) * ring + 1, ratio);
        append(layer.compressed, DataType::kUInt8, 1, page_count);
        append(layer.compressor_state, DataType::kFloat32, ring, state_rows);
        if (ratio == 4) {
            append(layer.indexer, DataType::kUInt8, 1, page_count);
            append(layer.indexer_state, DataType::kFloat32, ring, state_rows);
        }
    }
    std::sort(ranges.begin(), ranges.end());
    for (size_t i = 1; i < ranges.size(); ++i) {
        if (ranges[i - 1].second > ranges[i].first) {
            throw std::invalid_argument("V4 cache page clone stores alias");
        }
    }
    barrier.wait();
    try {
        for (const auto &store : stores) {
            for (const auto &[source, destination] : copies) {
                rt.memcpyD2D(store.data + destination * store.page_bytes,
                             store.data + source * store.page_bytes, store.page_bytes);
            }
        }
    } catch (...) {
        // A failed enqueue can follow successful enqueues on the same stream.
        drain_or_terminate(rt);
        throw;
    }
    rt.syncStream();
}

PagedCacheSnapshot::PagedCacheSnapshot(runtime::Runtime &rt,
                                       const std::vector<LayerPagedCache> &layers,
                                       const PagedCacheWriteSet &writes,
                                       runtime::Tensor *reusable_backup)
    : runtime_(rt) {
    rt.assertOwnerThread();
    if (rt.isGraphRecording()) {
        throw std::logic_error("DeepSeek V4 snapshot cannot run during graph recording");
    }
    if (layers.empty() || !writes.swa_slots || !writes.swa_slots->numel()) {
        throw std::invalid_argument("DeepSeek V4 snapshot requires caches and query rows");
    }
    const auto query_rows = writes.swa_slots->numel();
    const auto swa = written_rows(writes.swa_slots, query_rows, kV4PageSize);
    const auto c4 = written_rows(writes.c4_slots, query_rows, kV4PageSize / 4);
    const auto c128 = written_rows(writes.c128_slots, query_rows, kV4PageSize / 128);
    const auto c4_state = written_rows(writes.c4_state_slots, query_rows, 1);
    const auto c128_state = written_rows(writes.c128_state_slots, query_rows, 1);
    using runtime::DataType;
    size_t backup_bytes = 0;
    auto append = [&](const runtime::Tensor &storage, DataType type, const std::vector<size_t> &rows) {
        if (!storage || storage->device() != rt.device() || storage->dtype() != type ||
            storage->ndim() != 2 || !storage->is_contiguous() || !storage->size(1)) {
            throw std::invalid_argument("Invalid DeepSeek V4 snapshot storage");
        }
        if (!rows.empty() && rows.back() >= storage->size(0)) {
            throw std::out_of_range("DeepSeek V4 snapshot write exceeds cache storage");
        }
        const auto width = product(storage->size(1), runtime::dsize(type));
        for (size_t first = 0; first < rows.size();) {
            size_t end = first + 1;
            while (end < rows.size() && rows[end] == rows[end - 1] + 1) {
                ++end;
            }
            const auto bytes = product(end - first, width);
            if (backup_bytes > std::numeric_limits<size_t>::max() - bytes) {
                throw std::overflow_error("DeepSeek V4 snapshot exceeds addressable size");
            }
            slices_.push_back({storage, product(rows[first], width), backup_bytes, bytes});
            backup_bytes += bytes;
            first = end;
        }
    };

    for (const auto &layer : layers) {
        append(layer.swa, DataType::kUInt8, swa);
        switch (layer.compression_ratio) {
        case 0:
            break;
        case 4:
            append(layer.compressed, DataType::kUInt8, c4);
            append(layer.indexer, DataType::kUInt8, c4);
            append(layer.compressor_state, DataType::kFloat32, c4_state);
            append(layer.indexer_state, DataType::kFloat32, c4_state);
            break;
        case 128:
            append(layer.compressed, DataType::kUInt8, c128);
            append(layer.compressor_state, DataType::kFloat32, c128_state);
            break;
        default:
            throw std::invalid_argument("Invalid DeepSeek V4 snapshot compression ratio");
        }
    }

    if (backup_bytes) {
        if (reusable_backup && *reusable_backup && (*reusable_backup)->numel() >= backup_bytes) {
            backup_ = *reusable_backup;
        } else {
            backup_ = runtime::Tensor::empty(rt, {backup_bytes}, DataType::kUInt8);
            if (reusable_backup) {
                *reusable_backup = backup_;
            }
        }
        try {
            for (auto &slice : slices_) {
                rt.memcpyD2D(backup_->data() + slice.backup_offset,
                             slice.live->data() + slice.live_offset, slice.bytes);
            }
        } catch (...) {
            // The allocator may recycle this buffer as soon as the constructor unwinds.
            drain_or_terminate(rt);
            throw;
        }
    }
    active_ = true;
}

PagedCacheSnapshot::~PagedCacheSnapshot() noexcept {
    if (active_) {
        try {
            rollback();
        } catch (...) {
            std::terminate();
        }
    }
}

void PagedCacheSnapshot::require_active() const {
    runtime_.assertOwnerThread();
    if (!active_ || runtime_.isGraphRecording()) {
        throw std::logic_error("DeepSeek V4 snapshot requires an active eager transaction");
    }
}

void PagedCacheSnapshot::release() {
    runtime_.syncStream();
    backup_ = {};
    slices_.clear();
    active_ = false;
}

void PagedCacheSnapshot::commit() {
    require_active();
    release();
}

void PagedCacheSnapshot::rollback() {
    require_active();
    for (auto &slice : slices_) {
        runtime_.memcpyD2D(slice.live->data() + slice.live_offset,
                          backup_->data() + slice.backup_offset, slice.bytes);
    }
    release();
}

} // namespace infinilm::models::deepseek_v4
