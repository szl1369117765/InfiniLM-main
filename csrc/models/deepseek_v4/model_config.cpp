#include "model_config.hpp"

#include "models/models_registry.hpp"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace infinilm::models::deepseek_v4 {
namespace {

void normalize_geometry(config::ModelConfig &source) {
    auto &json = source.get_config_json();
    const auto head = source.get<int64_t>("head_dim");
    const auto rotary = source.get<int64_t>("qk_rope_head_dim");
    const auto layers = source.get<int64_t>("num_hidden_layers");
    if (head <= 0 || rotary <= 0 || rotary > head || rotary % 2 || layers <= 0) {
        throw std::invalid_argument("Invalid DeepSeek V4 head or layer geometry");
    }
    const auto ratios = source.get<std::vector<int64_t>>("compress_ratios");
    if (ratios.size() < static_cast<size_t>(layers)
        || !std::all_of(ratios.begin(), ratios.end(), [](auto x) {
            return x == 0 || x == 4 || x == 128;
        })) {
        throw std::invalid_argument("Invalid DeepSeek V4 compression ratios");
    }
    if (json.contains("qk_nope_head_dim")
        && source.get<int64_t>("qk_nope_head_dim") != head - rotary) {
        throw std::invalid_argument("DeepSeek V4 rotary and non-rotary dimensions disagree");
    }
    json["qk_nope_head_dim"] = head - rotary;
    if ((!json.contains("quantization_config") || json["quantization_config"].is_null())
        && json.contains("compression_config")) {
        json["quantization_config"] = json["compression_config"];
    }
}

} // namespace

std::shared_ptr<config::ModelConfig> create_deepseek_v4_model_config(
    std::shared_ptr<config::ModelConfig> source) {
    if (!source || source->get_or<std::string>("model_type", "") != "deepseek_v4") {
        throw std::invalid_argument("DeepSeek V4 config requires model_type=deepseek_v4");
    }
    normalize_geometry(*source);
    return source;
}

std::shared_ptr<config::ModelConfig> create_deepseek_v4_mtp_model_config(
    std::shared_ptr<config::ModelConfig> source) {
    if (!source || source->get_or<std::string>("model_type", "") != "deepseek_v4_mtp") {
        throw std::invalid_argument("Classic NextN config requires model_type=deepseek_v4_mtp");
    }
    const auto &json = source->get_config_json();
    for (auto it = json.begin(); it != json.end(); ++it) {
        if (it.key().compare(0, 7, "dspark_") == 0) {
            throw std::invalid_argument("Classic NextN does not support DSpark checkpoints");
        }
    }
    if ((json.contains("num_nextn_predict_layers")
         && source->get<int64_t>("num_nextn_predict_layers") != 1)
        || source->get<int64_t>("num_hidden_layers") != 1
        || source->get<int64_t>("num_hash_layers") != 0
        || source->get<std::vector<int64_t>>("compress_ratios") != std::vector<int64_t>{0}) {
        throw std::invalid_argument("Classic NextN requires one uncompressed, non-hash Draft layer");
    }
    normalize_geometry(*source);
    return source;
}

} // namespace infinilm::models::deepseek_v4
