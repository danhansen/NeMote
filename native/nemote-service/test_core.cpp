#include "core.h"
#include <iostream>
using namespace nemote;
static void check(bool condition) { if (!condition) throw std::runtime_error("Core check failed"); }
int main() {
    auto config = defaults("/home/test", "/opt/nemote", "/home/test/.local/share");
    validate(config);
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
