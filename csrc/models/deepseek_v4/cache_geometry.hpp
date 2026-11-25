#pragma once

#include <cstddef>

namespace infinilm::models::deepseek_v4 {

inline constexpr std::size_t kV4PageSize = 256;

constexpr std::size_t compression_window(std::size_t ratio) {
    return ratio == 4 ? 8 : 128;
}

// Keep one compression window of history when a short verify writes ahead.
constexpr std::size_t compressor_ring_rows(std::size_t ratio) {
    return 2 * compression_window(ratio);
}

static_assert(kV4PageSize % compressor_ring_rows(4) == 0);
static_assert(kV4PageSize % compressor_ring_rows(128) == 0);

} // namespace infinilm::models::deepseek_v4
