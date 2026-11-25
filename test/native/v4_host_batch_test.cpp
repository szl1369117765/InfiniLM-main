#include "engine/deepseek_v4_metadata.hpp"
#include "models/deepseek_v4/cache_geometry.hpp"

#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace {
using namespace infinilm::engine;
using namespace infinilm::runtime;

constexpr size_t page_size = infinilm::models::deepseek_v4::kV4PageSize;
const Device cpu{Device::Type::kCpu};

void require(bool condition, const char *message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void same(const Tensor &left, const Tensor &right) {
    require(left->shape() == right->shape() && left->dtype() == right->dtype(),
            "host metadata shape or dtype differs");
    if (left->numel()) {
        require(std::memcmp(left->data(), right->data(), left->numel() * sizeof(int32_t)) == 0,
                "host metadata bytes differ");
    }
}

template <class Error, class Callback>
void rejects(Callback callback) {
    try {
        callback();
    } catch (const Error &) {
        return;
    }
    throw std::runtime_error("invalid frozen V4 metadata accepted");
}

void compare_short_query(size_t query, bool classic_mtp_target) {
    int64_t positions[] = {255, 256, 257, 258, 259};
    int32_t offsets[] = {0, static_cast<int32_t>(query)};
    int32_t pages[] = {3, 1};
    int64_t slots[] = {3 * page_size + 255, page_size, page_size + 1,
                      page_size + 2, page_size + 3};
    const auto pos = Tensor::from_blob(positions, {query}, DataType::kInt64, cpu);
    const auto off = Tensor::from_blob(offsets, {2}, DataType::kInt32, cpu);
    const auto table = Tensor::from_blob(pages, {1, 2}, DataType::kInt32, cpu);
    const auto loc = Tensor::from_blob(slots, {query}, DataType::kInt64, cpu);
    const auto shared = DeepSeekV4HostBatch::prepare_short(pos, off, table, loc, query, 4, true, classic_mtp_target);
    const auto shadow = DeepSeekV4HostBatch::prepare_short(pos, off, table, loc, query, 4, false, classic_mtp_target);
    if (query > 4) {
        require(!shared && !shadow, "long queries must use the original path");
        return;
    }
    require(shared && shared->plans() && shadow && !shadow->plans(), "mode contract failed");

    const auto reference_attention = build_deepseek_v4_attention_metadata(
        table, loc, pos, off, std::nullopt, page_size, classic_mtp_target);
    const auto reference_c4 = build_deepseek_v4_compression_plan(reference_attention, pos, off, 4);
    const auto reference_c128 = build_deepseek_v4_compression_plan(reference_attention, pos, off, 128);
    // Position 255 is the last token on physical page 3.
    require(reinterpret_cast<const int32_t *>(reference_c4.state_slots->data())[0] ==
                (classic_mtp_target ? 63 : 31), "C4 raw ring does not use the selected page stride");
    require(reinterpret_cast<const int32_t *>(reference_c128.state_slots->data())[0] ==
                (classic_mtp_target ? 1023 : 511), "C128 raw ring does not use the selected page stride");

    // Mutating the caller's storage after preparation cannot change the
    // host plan, the worker's fallback plan, or the transfer inputs.
    const auto original_slot = slots[0];
    positions[0] = -1;
    offsets[1] = 0;
    pages[0] = -1;
    slots[0] = -1;
    for (const auto &frozen : {shared, shadow}) {
        require(reinterpret_cast<const int64_t *>(frozen->positions()->data())[0] == 255,
                "positions still alias the caller");
        require(reinterpret_cast<const int32_t *>(frozen->offsets()->data())[1] == static_cast<int32_t>(query),
                "offsets still alias the caller");
        require(reinterpret_cast<const int32_t *>(frozen->tables()->data())[0] == 3,
                "page table still aliases the caller");
        require(reinterpret_cast<const int64_t *>(frozen->slots()->data())[0] == original_slot,
                "slots still alias the caller");
    }
    const auto &plans = *shared->plans();
    const auto shadow_attention = build_deepseek_v4_attention_metadata(
        shadow->tables(), shadow->slots(), shadow->positions(), shadow->offsets(), std::nullopt,
        page_size, classic_mtp_target);
    const auto shadow_c4 = build_deepseek_v4_compression_plan(shadow_attention, shadow->positions(),
                                                              shadow->offsets(), 4);
    const auto shadow_c128 = build_deepseek_v4_compression_plan(shadow_attention, shadow->positions(),
                                                                shadow->offsets(), 128);
    same(plans.attention.raw_out_loc, reference_attention.raw_out_loc);
    same(plans.attention.c4_out_loc, reference_attention.c4_out_loc);
    same(plans.attention.c128_out_loc, reference_attention.c128_out_loc);
    same(shadow_attention.raw_out_loc, reference_attention.raw_out_loc);
    same(plans.c4.sources, reference_c4.sources);
    same(plans.c4.state_slots, reference_c4.state_slots);
    same(plans.c128.sources, reference_c128.sources);
    same(plans.c128.state_slots, reference_c128.state_slots);
    same(shadow_c4.sources, reference_c4.sources);
    same(shadow_c4.state_slots, reference_c4.state_slots);
    same(shadow_c128.sources, reference_c128.sources);
    same(shadow_c128.state_slots, reference_c128.state_slots);

    positions[0] = 255;
    offsets[1] = static_cast<int32_t>(query);
    pages[0] = 3;
    rejects<std::invalid_argument>([&] {
        DeepSeekV4HostBatch::prepare_short(pos, off, table, loc, query, 4, false);
    });
    slots[0] = original_slot;
    pages[0] = 4;
    rejects<std::out_of_range>([&] {
        DeepSeekV4HostBatch::prepare_short(pos, off, table, loc, query, 4, true);
    });
}

void compare_packed_target() {
    constexpr size_t rows = 9;
    int64_t positions[rows];
    int64_t slots[rows];
    int32_t offsets[] = {0, 2, 5, 9};
    int32_t pages[] = {3, 1, 5, 7, 8, 2};
    for (size_t owner = 0; owner < 3; ++owner) {
        for (auto row = offsets[owner]; row < offsets[owner + 1]; ++row) {
            const size_t position = page_size - 1 + row - offsets[owner];
            positions[row] = position;
            slots[row] = pages[owner * 2 + position / page_size] * page_size + position % page_size;
        }
    }
    const auto pos = Tensor::from_blob(positions, {rows}, DataType::kInt64, cpu);
    const auto loc = Tensor::from_blob(slots, {rows}, DataType::kInt64, cpu);
    const auto off = Tensor::from_blob(offsets, {4}, DataType::kInt32, cpu);
    const auto table = Tensor::from_blob(pages, {3, 2}, DataType::kInt32, cpu);
    const auto shared = DeepSeekV4HostBatch::prepare_short(pos, off, table, loc, rows, 9, true, true);
    require(shared && shared->plans(), "packed Target lost its shared plan");
    const auto reference = build_deepseek_v4_attention_metadata(table, loc, pos, off, std::nullopt, page_size, true);
    const auto c4 = build_deepseek_v4_compression_plan(reference, pos, off, 4);
    const auto c128 = build_deepseek_v4_compression_plan(reference, pos, off, 128);
    const auto &plans = *shared->plans();
    same(plans.attention.raw_out_loc, reference.raw_out_loc);
    same(plans.attention.c4_out_loc, reference.c4_out_loc);
    same(plans.attention.c128_out_loc, reference.c128_out_loc);
    same(plans.c4.sources, c4.sources);
    same(plans.c4.state_slots, c4.state_slots);
    same(plans.c128.sources, c128.sources);
    same(plans.c128.state_slots, c128.state_slots);
}
} // namespace

int main() {
    try {
        for (const size_t query : {1, 2, 3, 4, 5}) {
            compare_short_query(query, false);
            compare_short_query(query, true);
        }
        compare_packed_target();
    } catch (const std::exception &error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
    return 0;
}
