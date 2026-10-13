// Developer-only label probe for checkpoint layout parity; not installed.
#include "recognizer.h"
#include "pnc_model.h"
#include <ggml-backend.h>
#include <nlohmann/json.hpp>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 4) return 2; // CPU plugin directory, ASR GGUF, PnC GGUF.
    try {
        ggml_backend_load_all_from_path(argv[1]);
        nemo_speech::asr::RecognizerConfig config;
        config.backend.gpu = -1;
        config.backend.threads = 2;
        config.model.path = argv[2];
        config.batching.enabled = false;
        config.batching.state_arena_slots = 1;
        nemo_speech::asr::Recognizer recognizer(config);
        nemo_speech::asr::pnc::PncModel model(recognizer.model()->backend_manager(), argv[3], config.batching);
        std::string line;
        while (std::getline(std::cin, line)) {
            auto ids = nlohmann::json::parse(line).get<std::vector<int32_t>>();
            std::vector<int> punct, capit;
            model.infer(ids.data(), static_cast<int>(ids.size()), punct, capit);
            std::cout << nlohmann::json({{"punct", punct}, {"capit", capit}}).dump() << std::endl;
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
