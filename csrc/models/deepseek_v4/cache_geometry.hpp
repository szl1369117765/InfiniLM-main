#pragma once

#include <cstddef>

namespace infinilm::models::deepseek_v4 {

inline constexpr std::size_t kV4PageSize = 256;
inline constexpr char kClassicMtpTargetKey[] = "_infinilm_classic_mtp_target";

constexpr std::size_t compression_window(std::size_t ratio) {
    return ratio == 4 ? 8 : 128;
}

// Keep an extra compression window of history for classic MTP verification.
constexpr std::size_t compressor_ring_rows(std::size_t ratio, bool classic_mtp_target) {
    return (classic_mtp_target ? 2 : 1) * compression_window(ratio);
}

static_assert(kV4PageSize % compressor_ring_rows(4, true) == 0);
static_assert(kV4PageSize % compressor_ring_rows(128, true) == 0);

} // namespace infinilm::models::deepseek_v4
