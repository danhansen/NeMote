#include "core.h"
#include <iostream>
using namespace nemote;
static void check(bool condition) { if (!condition) throw std::runtime_error("Core check failed"); }
int main() {
    auto config = defaults("/home/test", "/opt/nemote", "/home/test/.local/share");
    validate(config);
    check(config["pnc"] == false && config["pnc_model_path"] == "");
    auto pnc_config = config;
    pnc_config["pnc"] = true;
    validate(pnc_config); // Missing models may be configured, but must be installed before Start.
    check(!pnc_ready(pnc_config));
    check(pnc_ready(config)); // Disabled PnC does not need a model.
    pnc_config["pnc_variant"] = "distilbert";
    validate(pnc_config);
    const auto automatic_pnc_args = worker_command(pnc_config);
    auto automatic_path = std::find(automatic_pnc_args.begin(), automatic_pnc_args.end(), "--pnc-model");
    check(automatic_path != automatic_pnc_args.end() && *++automatic_path ==
        "/home/test/.local/share/nemote/models/pnc/pnc-distilbert-en.q8_0.gguf");
    auto variant = std::find(automatic_pnc_args.begin(), automatic_pnc_args.end(), "--pnc-variant");
    check(variant != automatic_pnc_args.end() && *++variant == "distilbert");
    pnc_config["pnc_model_path"] = "/missing-pnc-model.gguf";
    bool pnc_rejected = false;
    try { validate(pnc_config); } catch (const std::invalid_argument&) { pnc_rejected = true; }
    check(pnc_rejected);
    pnc_config["pnc_model_path"] = __FILE__;
    validate(pnc_config); // Model parsing belongs to the SDK, not config validation.
    const auto pnc_args = worker_command(pnc_config);
    check(std::find(pnc_args.begin(), pnc_args.end(), "--pnc") != pnc_args.end());
    auto pnc_path = std::find(pnc_args.begin(), pnc_args.end(), "--pnc-model");
    check(pnc_path != pnc_args.end() && *++pnc_path == __FILE__);
    pnc_config["model_family"] = "multilingual";
    pnc_config["language"] = "es-US";
    pnc_rejected = false;
    try { validate(pnc_config); } catch (const std::invalid_argument&) { pnc_rejected = true; }
    check(pnc_rejected);
    pnc_config["language"] = "auto";
    pnc_rejected = false;
    try { validate(pnc_config); } catch (const std::invalid_argument&) { pnc_rejected = true; }
    check(pnc_rejected);
    pnc_config["model_family"] = "english";
    validate(pnc_config);
    auto migrated = migrate(config, {{"backend", "parakeet"}, {"model_profile", "fast"},
        {"worker_path", "/opt/wordpipe/bin/wordpipe-parakeet-worker"}, {"input_device", "cpal:3"}, {"itn", true}});
    check(migrated["backend"] == "nemo-speech" && migrated["model_profile"] == "nemo-q8");
    check(migrated["worker_path"] == "/opt/nemote/bin/nemote-nemo-worker" && migrated["input_device"] == "" && migrated["itn"] == true);
    auto renamed = migrate(config, {{"worker_path", "/opt/wordpipe/bin/wordpipe-nemo-worker"},
        {"model_installer_path", "/opt/wordpipe/bin/wordpipe-model-install"},
        {"model_root", "/data/wordpipe/models"}, {"itn", true}, {"endpoint_mode", "preview"}});
    check(renamed["worker_path"] == config["worker_path"] && renamed["model_installer_path"] == config["model_installer_path"]);
    check(renamed["model_root"] == "/data/wordpipe/models" && renamed["itn"] == true && renamed["endpoint_mode"] == "preview");
    auto custom = migrate(config, {{"worker_path", "/custom/worker"}, {"model_installer_path", "/custom/installer"}});
    check(custom["worker_path"] == "/custom/worker" && custom["model_installer_path"] == "/custom/installer");
    check(preset(config) == "nemo-q8-english");
    for (auto size : latencies()) {
        config["streaming_latency_ms"] = size; validate(config);
        auto args = worker_command(config);
        auto chunk = std::find(args.begin(), args.end(), "--chunk-samples");
        check(chunk != args.end() && *++chunk == std::to_string(size.get<unsigned>() * 16));
    }
    config["streaming_latency_ms"] = 240;
    bool rejected = false; try { validate(config); } catch (const std::invalid_argument&) { rejected = true; } check(rejected);
    check(punctuation("hello comma world new paragraph goodbye full stop", false) == "hello, world\n\ngoodbye.");
    check(punctuation("hello question", true) == "hello");
    check(punctuation("hello question mark", false) == "hello?");
    std::cout << "Service config, migration, chunk sizes and punctuation passed\n";
}
