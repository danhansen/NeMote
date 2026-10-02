#pragma once
#include <nlohmann/json.hpp>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace wordpipe {
using Json = nlohmann::json;
namespace fs = std::filesystem;
inline constexpr const char* bus_name = "dev.wordpipe.Service";
inline constexpr const char* object_path = "/dev/wordpipe/Service";
inline constexpr const char* interface_name = "dev.wordpipe.Service1";
inline Json latencies() { return Json::array({80, 160, 560, 1120}); }
inline std::string trim(std::string text) {
    auto start = text.find_first_not_of(" \t\r\n");
    return start == std::string::npos ? "" : text.substr(start, text.find_last_not_of(" \t\r\n") - start + 1);
}
inline Json defaults(const fs::path&, const fs::path& prefix, const fs::path& data_home) {
    return {{"backend", "nemo-speech"}, {"model_profile", "nemo-q8"}, {"model_family", "english"},
        {"streaming_latency_ms", 560u}, {"input_device", ""}, {"language", "en-US"},
        {"shortcut", "<Control><Alt>space"}, {"model_root", (data_home / "wordpipe/models").string()},
        {"worker_path", (prefix / "bin/wordpipe-nemo-worker").string()},
        {"model_installer_path", (prefix / "bin/wordpipe-model-install").string()},
        {"sample_rate", 16000u}, {"num_threads", 2u}, {"spoken_punctuation", true}, {"itn", false},
        {"phrase_boosting", false}, {"boost_phrases", ""}, {"boost_tokenizer_path", ""},
        {"vad_filtering", false}, {"vad_model_path", ""}, {"endpoint_mode", "disabled"},
        {"insert_partials", true}, {"stream_insert_delay_ms", 0u}, {"show_overlay", true}};
}
inline std::string preset(const Json& config) {
    return config.at("model_family") == "english" ? "nemo-q8-english" : "nemo-q8";
}
inline fs::path runtime(const Json& config, const std::string& family = "") {
    const auto selected = family.empty() ? config.at("model_family").get<std::string>() : family;
    return fs::path(config.at("model_root").get<std::string>()) /
        (selected == "english" ? "nemotron-nemo-en-q8" : "nemotron-nemo-q8");
}
inline bool installed(const Json& config, const std::string& family = "") {
    auto path = runtime(config, family) / "model.gguf";
    auto selected = family.empty() ? config.at("model_family").get<std::string>() : family;
    std::error_code error;
    if (fs::file_size(path, error) != (selected == "english" ? 699872960u : 742090464u) || error) return false;
    char header[4]{};
    std::ifstream input(path, std::ios::binary);
    input.read(header, 4);
    return input && std::string(header, 4) == "GGUF";
}
inline bool supported_language(const std::string& language) {
    static const std::vector<std::string> codes = {"en-US", "en-GB", "auto", "es-US", "es-ES", "fr-FR", "fr-CA",
        "de-DE", "it-IT", "pt-BR", "pt-PT", "nl-NL", "tr-TR", "ru-RU", "ar-AR", "hi-IN", "ja-JP", "ko-KR",
        "vi-VN", "uk-UA", "pl-PL", "sv-SE", "cs-CZ", "nb-NO", "da-DK", "bg-BG", "fi-FI", "hr-HR", "sk-SK",
        "zh-CN", "zh-TW", "hu-HU", "ro-RO", "et-EE", "el-GR", "lt-LT", "lv-LV", "bn-IN", "id-ID", "ms-MY", "th-TH"};
    return std::find(codes.begin(), codes.end(), language) != codes.end();
}
inline void validate(const Json& config) {
    auto require = [](bool valid, const char* message) { if (!valid) throw std::invalid_argument(message); };
    require(config.at("backend") == "nemo-speech", "Only NeMo-Speech.cpp is supported");
    require(config.at("model_profile") == "nemo-q8", "Only the NeMo Q8 model profile is supported");
    require(config.at("model_family") == "english" || config.at("model_family") == "multilingual", "Unknown model family");
    auto sizes = latencies();
    require(std::find(sizes.begin(), sizes.end(), config.at("streaming_latency_ms")) != sizes.end(), "Chunk size must be 80, 160, 560, or 1120 ms");
    require(config.at("sample_rate") == 16000, "Sample rate must be 16000 Hz");
    require(config.at("num_threads").get<int64_t>() > 0 && config.at("num_threads").get<int64_t>() <= 256, "Thread count must be 1–256");
    require(config.at("stream_insert_delay_ms").get<int64_t>() >= 0 && config.at("stream_insert_delay_ms").get<int64_t>() <= 60000, "Insertion delay must be 0–60000 ms");
    auto language = config.at("language").get<std::string>();
    require(supported_language(language), "Unknown language");
    require(config.at("model_family") != "english" || language == "auto" || language.rfind("en-", 0) == 0, "English model requires English or Auto language");
    auto mode = config.at("endpoint_mode").get<std::string>();
    require(mode == "disabled" || mode == "preview" || mode == "commit", "Endpoint mode must be disabled, preview, or commit");
    auto phrases = config.at("boost_phrases").get<std::string>();
    require(phrases.size() <= 16384, "Boost phrases exceed 16 KiB");
    require(!config.at("phrase_boosting").get<bool>() || !trim(phrases).empty(), "Enter names or phrases before enabling boosting");
    for (const auto* key : {"model_root", "worker_path", "model_installer_path"})
        require(!trim(config.at(key).get<std::string>()).empty(), "Model and executable paths must not be empty");
    if (config.at("vad_filtering").get<bool>() && config.at("vad_model_path") != "")
        require(fs::is_regular_file(config.at("vad_model_path").get<std::string>()), "VAD model override does not exist");
}
inline Json migrate(Json config, const Json& saved) {
    if (!saved.is_object()) throw std::invalid_argument("Service config must be a JSON object");
    for (auto item = config.begin(); item != config.end(); ++item) {
        if (!saved.contains(item.key()) || saved[item.key()].is_null()) continue;
        const auto& value = saved[item.key()];
        if ((item->is_number() && !value.is_number_integer()) ||
            (item->is_boolean() && !value.is_boolean()) || (item->is_string() && !value.is_string()))
            throw std::invalid_argument("Invalid config type: " + item.key());
        *item = value;
    }
    // CPAL indexes and the old worker executable are not valid in miniaudio.
    bool old = config["backend"] == "parakeet" || config["model_profile"] == "fast" || config["model_profile"] == "compact";
    config["backend"] = "nemo-speech";
    config["model_profile"] = "nemo-q8";
    config["sample_rate"] = 16000u;
    if (old) config["input_device"] = "";
    if (fs::path(config["worker_path"].get<std::string>()).filename() == "wordpipe-parakeet-worker")
        config["worker_path"] = (fs::path(config["worker_path"].get<std::string>()).parent_path() / "wordpipe-nemo-worker").string();
    if (old && config["model_family"] == "english" && config["language"] != "auto" &&
        config["language"].get<std::string>().rfind("en-", 0) != 0) config["language"] = "en-US";
    validate(config);
    return config;
}
inline std::vector<std::string> worker_command(const Json& config) {
    std::vector<std::string> command = {config.at("worker_path"), "--model-dir", runtime(config).string(),
        "--model-family", config.at("model_family"), "--num-threads", std::to_string(config.at("num_threads").get<unsigned>()),
        "--sample-rate", "16000", "--chunk-samples", std::to_string(config.at("streaming_latency_ms").get<unsigned>() * 16),
        "--language", config.at("language"), "--device", "cpu"};
    auto add = [&](const char* key, const char* argument) { if (config.at(key) != "") { command.push_back(argument); command.push_back(config.at(key)); } };
    add("input_device", "--input-device");
    if (config.at("itn")) command.push_back("--itn");
    command.insert(command.end(), {"--endpoint-mode", config.at("endpoint_mode")});
    if (config.at("phrase_boosting")) {
        command.insert(command.end(), {"--phrase-boosting", "--boost-phrases", config.at("boost_phrases")});
        add("boost_tokenizer_path", "--boost-tokenizer");
    }
    if (config.at("vad_filtering")) { command.push_back("--vad-filtering"); add("vad_model_path", "--vad-model"); }
    auto helper = fs::path(config.at("model_installer_path").get<std::string>()).parent_path() / "wordpipe-companion-install";
    if (fs::is_regular_file(helper)) command.insert(command.end(), {"--companion-installer", helper.string()});
    return command;
}
inline std::string punctuation(const std::string& text, bool partial) {
    std::istringstream input(text);
    std::vector<std::string> words;
    for (std::string word; input >> word;) words.push_back(word);
    auto lower = [](std::string word) { for (auto& c : word) if (c >= 'A' && c <= 'Z') c += 'a' - 'A'; return word; };
    if (partial && !words.empty()) {
        auto tail = lower(words.back());
        if (tail == "full" || tail == "question" || tail == "exclamation" || tail == "new") words.pop_back();
    }
    std::string output;
    for (size_t i = 0; i < words.size(); ++i) {
        auto word = lower(words[i]);
        auto second = i + 1 < words.size() ? lower(words[i + 1]) : "";
        std::string mark;
        if (word == "comma") mark = ",";
        else if (word == "period") mark = ".";
        else if (word == "colon") mark = ":";
        else if (word == "semicolon") mark = ";";
        else if (word == "newline") mark = "\n";
        else if (word == "full" && second == "stop") { mark = "."; ++i; }
        else if (word == "question" && second == "mark") { mark = "?"; ++i; }
        else if (word == "exclamation" && (second == "mark" || second == "point")) { mark = "!"; ++i; }
        else if (word == "new" && (second == "line" || second == "paragraph")) { mark = second == "line" ? "\n" : "\n\n"; ++i; }
        if (!mark.empty()) { while (!output.empty() && output.back() == ' ') output.pop_back(); output += mark; if (mark[0] != '\n') output += ' '; }
        else { if (!output.empty() && output.back() != ' ' && output.back() != '\n') output += ' '; output += words[i] + ' '; }
    }
    while (!output.empty() && output.back() == ' ') output.pop_back();
    return output;
}
} // namespace wordpipe
