#include "engine/rank_barrier.hpp"
#include "models/deepseek_v4/paged_cache.hpp"
#include "runtime/runtime.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <future>
#include <stdexcept>
#include <string>

namespace {
using infinilm::engine::RankBarrier;
using namespace infinilm::models::deepseek_v4;
using namespace infinilm::runtime;

constexpr size_t pages = 5;
constexpr int padding = 0xed;
const std::vector<std::pair<size_t, size_t>> copies{{0, 2}, {1, 3}, {0, 4}};

void require(bool condition, const char *description) {
    if (!condition) {
        throw std::runtime_error(description);
    }
}

template <class Error>
void rejects(const std::vector<std::pair<size_t, size_t>> &values) {
    try {
        validate_cache_page_copies(values, pages);
    } catch (const Error &) {
        return;
    }
    throw std::runtime_error("Invalid copy list was accepted");
}

template <class Function>
void visit_stores(std::vector<LayerPagedCache> &layers, Function visit) {
    int store = 0;
    for (auto &layer : layers) {
        visit(layer.swa, 1, ++store);
        if (layer.compression_ratio != 0) {
            visit(layer.compressed, 1, ++store);
            const size_t raw_rows = layer.compression_ratio == 4 ? 16 : 256;
            visit(layer.compressor_state, raw_rows, ++store);
            if (layer.compression_ratio == 4) {
                visit(layer.indexer, 1, ++store);
                visit(layer.indexer_state, raw_rows, ++store);
            }
        }
    }
}

std::vector<LayerPagedCache> make_fixture(Runtime &runtime) {
    std::vector<LayerPagedCache> layers;
    for (const size_t ratio : {0, 4, 128}) {
        layers.push_back(allocate_layer_paged_cache(runtime, pages, pages, ratio));
    }
    visit_stores(layers, [](Tensor &storage, size_t rows, int store) {
        const auto bytes = rows * storage->size(1) * storage->element_size();
        std::memset(storage->data(), padding, storage->nbytes());
        for (size_t page = 0; page < pages; ++page) {
            std::memset(storage->data() + page * bytes, 16 * store + page, bytes);
        }
    });
    return layers;
}

void expect_pages(std::vector<LayerPagedCache> &layers, const std::vector<int> &source_pages) {
    visit_stores(layers, [&](Tensor &storage, size_t rows, int store) {
        const auto *data = storage->data();
        const auto bytes = rows * storage->size(1) * storage->element_size();
        for (size_t page = 0; page < pages; ++page) {
            const auto value = static_cast<std::byte>(16 * store + source_pages[page]);
            require(std::all_of(data + page * bytes, data + (page + 1) * bytes,
                                [=](std::byte elem) { return elem == value; }),
                    "Physical page does not have the expected source bytes");
        }
        require(std::all_of(data + pages * bytes, data + storage->nbytes(),
                            [](std::byte elem) { return elem == std::byte{padding}; }),
                "Copy modified the raw-ring padding");
    });
}

void test_clone() {
    validate_cache_page_copies(copies, pages);
    rejects<std::invalid_argument>({});
    rejects<std::invalid_argument>({{0, 0}});
    rejects<std::invalid_argument>({{0, 2}, {1, 2}});
    rejects<std::invalid_argument>({{0, 2}, {2, 3}});
    rejects<std::out_of_range>({{0, pages}});
    rejects<std::out_of_range>({{pages, 0}});

    const auto runtime = Runtime::create(Device{Device::Type::kCpu, 0});
    runtime->activate();
    auto layers = make_fixture(*runtime);
    RankBarrier barrier(1);
    clone_cache_pages(*runtime, layers, pages, copies, barrier);
    expect_pages(layers, {0, 1, 0, 1, 0});
}

void test_invalid_rank_releases_peer() {
    RankBarrier barrier(2);
    auto rank = [&](bool alias) {
        try {
            const auto runtime = Runtime::create(Device{Device::Type::kCpu, 0});
            runtime->activate();
            auto layers = make_fixture(*runtime);
            if (alias) {
                layers[1].swa = layers[0].swa;
            }
            try {
                clone_cache_pages(*runtime, layers, pages, copies, barrier);
            } catch (const std::invalid_argument &error) {
                // Equivalent to RankWorker::fail after local preflight throws.
                barrier.abort(std::current_exception());
                require(std::string(error.what()) == "V4 cache page clone stores alias",
                        "The alias error was not propagated to both ranks");
                if (!alias) {
                    expect_pages(layers, {0, 1, 2, 3, 4});
                }
                return;
            }
            throw std::runtime_error("Copy ran despite an invalid rank's preflight");
        } catch (...) {
            barrier.abort(std::current_exception());
            throw;
        }
    };
    auto good = std::async(std::launch::async, rank, false);
    auto bad = std::async(std::launch::async, rank, true);
    bad.get();
    good.get();
}
} // namespace

int main() {
    try {
        test_clone();
        test_invalid_rank_releases_peer();
    } catch (const std::exception &error) {
        std::fprintf(stderr, "v4_cache_page_clone_test: %s\n", error.what());
        return 1;
    }
    std::puts("v4_cache_page_clone_test: PASS");
    return 0;
}
