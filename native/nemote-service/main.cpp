#include "core.h"
#include "interface.h"
#include <gio/gio.h>
#include <glib-unix.h>
#include <csignal>
#include <iostream>
#include <memory>
#include <sys/prctl.h>
#include <unistd.h>

using namespace nemote;

static GVariant* to_variant(const Json& value, const std::string& key = "") {
    if (value.is_object()) {
        GVariantBuilder builder;
        g_variant_builder_init(&builder, G_VARIANT_TYPE_VARDICT);
        for (auto item = value.begin(); item != value.end(); ++item)
            if (!item->is_null()) g_variant_builder_add(&builder, "{sv}", item.key().c_str(), to_variant(*item, item.key()));
        return g_variant_builder_end(&builder);
    }
    if (value.is_array()) {
        GVariantBuilder builder;
        g_variant_builder_init(&builder, G_VARIANT_TYPE("au"));
        for (const auto& item : value) g_variant_builder_add(&builder, "u", item.get<guint32>());
        return g_variant_builder_end(&builder);
    }
    if (value.is_boolean()) return g_variant_new_boolean(value.get<bool>());
    if (value.is_string()) return g_variant_new_string(value.get_ref<const std::string&>().c_str());
    if (value.is_number_float()) return g_variant_new_double(value.get<double>());
    if (key == "streaming_latency_ms" || key == "sample_rate" || key == "num_threads" || key == "stream_insert_delay_ms" || key == "index")
        return g_variant_new_uint32(value.get<guint32>());
    if (value.is_number_unsigned() || (value.is_number_integer() && value.get<int64_t>() >= 0)) return g_variant_new_uint64(value.get<guint64>());
    return g_variant_new_int64(value.get<gint64>());
}

static Json from_variant(GVariant* value) {
    if (g_variant_is_of_type(value, G_VARIANT_TYPE_VARIANT)) {
        auto* inner = g_variant_get_variant(value);
        auto result = from_variant(inner);
        g_variant_unref(inner);
        return result;
    }
    if (g_variant_is_of_type(value, G_VARIANT_TYPE_STRING)) return g_variant_get_string(value, nullptr);
    if (g_variant_is_of_type(value, G_VARIANT_TYPE_BOOLEAN)) return bool(g_variant_get_boolean(value));
    if (g_variant_is_of_type(value, G_VARIANT_TYPE_UINT32)) return g_variant_get_uint32(value);
    if (g_variant_is_of_type(value, G_VARIANT_TYPE_UINT64)) return g_variant_get_uint64(value);
    if (g_variant_is_of_type(value, G_VARIANT_TYPE_INT32)) return g_variant_get_int32(value);
    if (g_variant_is_of_type(value, G_VARIANT_TYPE_VARDICT)) {
        Json result = Json::object();
        GVariantIter iterator;
        g_variant_iter_init(&iterator, value);
        const gchar* key;
        GVariant* entry;
        while (g_variant_iter_next(&iterator, "{&sv}", &key, &entry)) {
            try { result[key] = from_variant(entry); } catch (...) { g_variant_unref(entry); throw; }
            g_variant_unref(entry);
        }
        return result;
    }
    throw std::invalid_argument("Unsupported option type");
}

static GVariant* list_variant(const Json& rows) {
    GVariantBuilder builder;
    g_variant_builder_init(&builder, G_VARIANT_TYPE("aa{sv}"));
    for (const auto& row : rows) g_variant_builder_add_value(&builder, to_variant(row));
    return g_variant_builder_end(&builder);
}

class Service;
struct Child {
    Service* owner;
    GSubprocess* process = nullptr;
    GDataInputStream* out = nullptr;
    GDataInputStream* err = nullptr;
    enum Kind { Worker, Installer, Devices } kind;
    std::string profile;
    pid_t pid = 0;
    std::string error;
    bool stdout_done = false, stderr_done = false, waited = false;
    bool data_error = false;
    guint timeout = 0;
    Json devices = Json::array();
    GDBusMethodInvocation* invocation = nullptr;
    ~Child() {
        if (invocation) g_object_unref(invocation);
        g_clear_object(&out); g_clear_object(&err); g_clear_object(&process);
    }
};

class Service {
public:
    GMainLoop* loop = nullptr;
    GDBusConnection* bus = nullptr;
    Json config;
    Json default_config;
    Json metrics = Json::object(), progress = Json::object();
    fs::path config_path;
    bool listening = false, stopping = false, loading = false, loaded = false, shutting_down = false;
    guint64 session_id = 0, seq = 0;
    std::string partial, committed, last_error, shell;
    std::shared_ptr<Child> worker, installer;
    std::vector<std::weak_ptr<Child>> children;

    explicit Service(const fs::path& path) : config_path(path) {
        fs::path home = g_get_home_dir();
        fs::path prefix = fs::canonical("/proc/self/exe").parent_path().parent_path();
        auto base = defaults(home, prefix, g_get_user_data_dir());
        default_config = base;
        config = base;
        auto saved_path = path;
        auto canonical_path = fs::path(g_get_user_config_dir()) / "nemote/service.json";
        auto legacy_path = fs::path(g_get_user_config_dir()) / "wordpipe/service.json";
        if (path == canonical_path && !fs::exists(path) && fs::is_regular_file(legacy_path)) saved_path = legacy_path;
        if (fs::exists(saved_path)) { std::ifstream input(saved_path); Json saved; input >> saved; config = migrate(base, saved); }
        if (!installed(config)) {
            auto other = config["model_family"] == "english" ? "multilingual" : "english";
            if (installed(config, other)) { config["model_family"] = other; config["language"] = "en-US"; }
        }
        validate(config);
    }
    ~Service() { shutdown_children(); }
    Json config_view() const { auto result = config; result["model_preset"] = preset(config); result["supported_streaming_latencies_ms"] = latencies(); return result; }
    Json state() const {
        return {{"listening", listening}, {"stopping", stopping}, {"loading_model", loading}, {"model_loaded", loaded},
            {"installing", bool(installer)}, {"installing_profile", installer ? installer->profile : ""},
            {"session_id", session_id}, {"seq", seq}, {"backend", "nemo-speech"}, {"model_profile", "nemo-q8"},
            {"model_family", config.at("model_family")}, {"model_preset", preset(config)}, {"input_device", config.at("input_device")},
            {"partial_text", partial}, {"last_commit_text", committed}, {"selected_runtime_dir", runtime(config).string()},
            {"selected_model_installed", installed(config)}, {"last_error", last_error}, {"last_metrics", metrics}, {"last_install_progress", progress}};
    }
    void signal(const char* name, GVariant* args) {
        GError* error = nullptr;
        if (!g_dbus_connection_emit_signal(bus, nullptr, object_path, interface_name, name, args, &error)) {
            std::cerr << "nemote-service: signal " << name << ": " << error->message << '\n'; g_error_free(error);
        }
    }
    void changed() { signal("StateChanged", g_variant_new("(@a{sv})", to_variant(state()))); }
    void fail(const std::string& message) { last_error = message; signal("Error", g_variant_new("(s)", message.c_str())); changed(); }
    void install_progress(Json event, const std::string& profile) {
        event["profile"] = profile; progress = event;
        signal("InstallProgress", g_variant_new("(s@a{sv})", profile.c_str(), to_variant(progress)));
    }
    void persist(const Json& next) {
        fs::create_directories(config_path.parent_path());
        auto path = config_path.string();
        auto content = next.dump(2) + '\n';
        GError* error = nullptr;
        // GLib writes to a sibling temporary file and renames it atomically.
        if (!g_file_set_contents_full(path.c_str(), content.data(), content.size(), G_FILE_SET_CONTENTS_CONSISTENT, 0600, &error)) {
            std::string message = error->message; g_error_free(error); throw std::runtime_error(message);
        }
    }
    void update(Json next, bool restart) {
        validate(next);
        if (next == config) return;
        persist(next);
        if (restart) stop_worker();
        config = std::move(next);
        signal("ConfigChanged", g_variant_new("(@a{sv})", to_variant(config_view()))); changed();
    }
    void normalize_paths(Json& next) const {
        for (const auto* key : {"model_root", "worker_path", "model_installer_path", "boost_tokenizer_path", "vad_model_path"}) {
            auto value = trim(next.at(key).get<std::string>());
            if (value.rfind("~/", 0) == 0) value = (fs::path(g_get_home_dir()) / value.substr(2)).string();
            if (value.empty() && default_config.at(key) != "") value = default_config.at(key);
            next[key] = value;
        }
    }
    bool busy() const { return listening || stopping || loading || bool(installer); }
    void require_idle() const { if (busy()) throw std::runtime_error("Stop dictation and wait for loading/installation before changing runtime settings"); }
    static void child_setup(gpointer) {
        setpgid(0, 0);
        prctl(PR_SET_PDEATHSIG, SIGKILL);
    }
    static void kill_child(const std::shared_ptr<Child>& child) {
        if (!child || child->waited) return;
        if (child->pid > 0) kill(-child->pid, SIGKILL);
        g_subprocess_force_exit(child->process);
    }
    void shutdown_children() {
        for (const auto& entry : children) if (auto child = entry.lock()) kill_child(child);
        worker.reset(); installer.reset();
    }
    void stop_worker() {
        auto old = std::move(worker); kill_child(old);
        bool active = listening || stopping;
        listening = stopping = loading = loaded = false; partial.clear();
        if (active && bus) signal("SessionStopped", g_variant_new("(t)", session_id));
    }
    static void read_line(const std::shared_ptr<Child>& child, bool stderr_stream);
    static void finish_if_ready(const std::shared_ptr<Child>& child);
    std::shared_ptr<Child> spawn(const std::vector<std::string>& args, Child::Kind kind, const std::string& profile = "") {
        std::vector<const char*> argv; for (const auto& arg : args) argv.push_back(arg.c_str()); argv.push_back(nullptr);
        auto flags = GSubprocessFlags(G_SUBPROCESS_FLAGS_STDOUT_PIPE | G_SUBPROCESS_FLAGS_STDERR_PIPE |
            (kind == Child::Worker ? G_SUBPROCESS_FLAGS_STDIN_PIPE : G_SUBPROCESS_FLAGS_NONE));
        auto* launcher = g_subprocess_launcher_new(flags);
        g_subprocess_launcher_set_child_setup(launcher, child_setup, nullptr, nullptr);
        GError* error = nullptr;
        auto* process = g_subprocess_launcher_spawnv(launcher, argv.data(), &error);
        g_object_unref(launcher);
        if (!process) { std::string message = error->message; g_error_free(error); throw std::runtime_error(message); }
        auto child = std::make_shared<Child>(); child->owner = this; child->process = process; child->kind = kind; child->profile = profile;
        child->pid = std::strtol(g_subprocess_get_identifier(process), nullptr, 10);
        child->out = g_data_input_stream_new(g_subprocess_get_stdout_pipe(process));
        child->err = g_data_input_stream_new(g_subprocess_get_stderr_pipe(process));
        children.erase(std::remove_if(children.begin(), children.end(), [](const auto& entry) { return entry.expired(); }), children.end());
        children.push_back(child);
        if (kind == Child::Devices) child->timeout = g_timeout_add_seconds_full(G_PRIORITY_DEFAULT, 10,
            [](gpointer data) -> gboolean {
                auto child = *static_cast<std::shared_ptr<Child>*>(data);
                child->timeout = 0; child->error = "Input-device enumeration timed out"; child->data_error = true;
                kill_child(child); return G_SOURCE_REMOVE;
            }, new std::shared_ptr<Child>(child), [](gpointer data) { delete static_cast<std::shared_ptr<Child>*>(data); });
        read_line(child, false); read_line(child, true);
        g_subprocess_wait_async(process, nullptr, [](GObject* process, GAsyncResult* result, gpointer data) {
            std::unique_ptr<std::shared_ptr<Child>> ref(static_cast<std::shared_ptr<Child>*>(data));
            GError* error = nullptr;
            if (!g_subprocess_wait_finish(G_SUBPROCESS(process), result, &error)) {
                (*ref)->error = error->message; g_error_free(error);
            }
            (*ref)->waited = true; finish_if_ready(*ref);
        }, new std::shared_ptr<Child>(child));
        return child;
    }
    void send(const Json& message) {
        if (!worker) throw std::runtime_error("ASR worker is not running");
        auto line = message.dump() + '\n'; GError* error = nullptr;
        if (!g_output_stream_write_all(g_subprocess_get_stdin_pipe(worker->process), line.data(), line.size(), nullptr, nullptr, &error)) {
            std::string message = error->message; g_error_free(error); throw std::runtime_error(message);
        }
    }
    void start() {
        if (stopping || installer) throw std::runtime_error("Wait for dictation to stop or model installation to finish");
        if (listening) return;
        if (!worker) {
            if (!installed(config)) throw std::runtime_error("NeMo model is not installed at " + runtime(config).string());
            worker = spawn(worker_command(config), Child::Worker); loading = true; loaded = false;
        }
        session_id = std::max(session_id + 1, guint64(g_get_real_time())); seq = 0;
        partial.clear(); committed.clear(); last_error.clear(); listening = true;
        send({{"command", "set_language"}, {"language", config["language"]}});
        send({{"command", "start"}});
        signal("SessionStarted", g_variant_new("(t)", session_id)); changed();
    }
    void stop() {
        if (stopping) return;
        listening = false;
        if (worker) { stopping = true; send({{"command", "stop"}}); }
        else signal("SessionStopped", g_variant_new("(t)", session_id));
        changed();
    }
    void event(const Json& value) {
        auto name = value.value("event", "");
        if (name == "loading_model") { loading = true; loaded = false; changed(); }
        else if (name == "model_loaded" || name == "ready") {
            loading = false; loaded = true;
            if (value.contains("data")) metrics = value["data"];
            changed(); signal("Metrics", g_variant_new("(@a{sv})", to_variant(metrics)));
        } else if (name == "listening") { if (!stopping) listening = true; changed(); }
        else if (name == "partial" || name == "commit") {
            if (!listening && !stopping) return; // Never insert stale output after a session ends.
            auto text = value.value("text", "");
            if (config["spoken_punctuation"]) text = punctuation(text, name == "partial");
            ++seq; metrics = value.value("data", Json::object());
            if (name == "partial") {
                signal("Partial", g_variant_new("(tts)", session_id, seq, text.c_str()));
                if (text.rfind(partial, 0) == 0 && text.size() > partial.size())
                    signal("TextDelta", g_variant_new("(tts)", session_id, seq, text.substr(partial.size()).c_str()));
                partial = text;
            } else {
                committed = text;
                if (!text.empty()) signal("Commit", g_variant_new("(tts)", session_id, seq, text.c_str()));
            }
            signal("Metrics", g_variant_new("(@a{sv})", to_variant(metrics)));
        } else if (name == "stats") { metrics = value.value("data", Json::object()); signal("Metrics", g_variant_new("(@a{sv})", to_variant(metrics))); }
        else if (name == "stopped") { listening = stopping = false; signal("SessionStopped", g_variant_new("(t)", session_id)); changed(); }
        else if (name == "error") {
            // Capture errors leave the model alive, but the session must not stay stuck.
            listening = stopping = loading = false;
            signal("SessionStopped", g_variant_new("(t)", session_id)); fail(value.value("message", "ASR worker error"));
        }
    }
    void shutdown() {
        if (shutting_down) return;
        shutting_down = true; stop_worker(); shutdown_children(); changed();
        g_main_loop_quit(loop);
    }
    Json profiles() const {
        Json rows = Json::array();
        for (const auto* family : {"english", "multilingual"}) {
            bool english = std::string(family) == "english";
            rows.push_back({{"id", english ? "nemo-q8-english" : "nemo-q8"}, {"title", english ? "English (lower WER)" : "Multilingual"},
                {"description", "Nemotron streaming model with optional inverse text normalization"}, {"model_profile", "nemo-q8"},
                {"model_family", family}, {"runtime_dir", runtime(config, family).string()}, {"installed", installed(config, family)}});
        }
        return rows;
    }
    void method(const std::string& name, GVariant* params, const char* sender, GDBusMethodInvocation* invocation) {
        if (shutting_down) throw std::runtime_error("Service is shutting down");
        auto string_arg = [&] { const gchar* value; g_variant_get(params, "(&s)", &value); return std::string(value); };
        if (name == "GetState") g_dbus_method_invocation_return_value(invocation, g_variant_new("(@a{sv})", to_variant(state())));
        else if (name == "GetConfig") g_dbus_method_invocation_return_value(invocation, g_variant_new("(@a{sv})", to_variant(config_view())));
        else if (name == "ListBackends") g_dbus_method_invocation_return_value(invocation, g_variant_new("(@aa{sv})", list_variant(Json::array({{{"id", "nemo-speech"}, {"title", "NeMo-Speech.cpp"}, {"description", "Native C++ Nemotron streaming"}}}))));
        else if (name == "ListModelProfiles") g_dbus_method_invocation_return_value(invocation, g_variant_new("(@aa{sv})", list_variant(profiles())));
        else if (name == "ListInputDevices") {
            auto child = spawn({config["worker_path"], "--list-input-devices"}, Child::Devices);
            child->invocation = G_DBUS_METHOD_INVOCATION(g_object_ref(invocation));
        } else {
            if (name == "RegisterShellClient") shell = sender;
            else if (name == "Start") start();
            else if (name == "Stop") stop();
            else if (name == "Toggle") { if (listening || stopping) stop(); else start(); }
            else if (name == "Shutdown") shutdown();
            else if (name == "SetBackend") {
                if (string_arg() != "nemo-speech") throw std::invalid_argument("Only NeMo-Speech.cpp is supported");
            } else if (name == "SetModelProfile") {
                auto id = string_arg(); require_idle();
                if (id != "nemo-q8" && id != "nemo-q8-english") throw std::invalid_argument("Unknown NeMo model preset");
                auto next = config; next["model_family"] = id == "nemo-q8" ? "multilingual" : "english";
                if (!installed(next)) throw std::runtime_error("Install the model before selecting it");
                if (next["model_family"] == "english" && next["language"] != "auto" && next["language"].get<std::string>().rfind("en-", 0) != 0) next["language"] = "en-US";
                update(next, true);
            } else if (name == "SetInputDevice") { require_idle(); auto next = config; next["input_device"] = string_arg(); update(next, true); }
            else if (name == "SetShortcut") { auto next = config; next["shortcut"] = string_arg(); update(next, false); }
            else if (name == "SetInsertionOptions" || name == "SetRuntimeOptions") {
                auto* dict = g_variant_get_child_value(params, 0);
                Json options;
                try { options = from_variant(dict); } catch (...) { g_variant_unref(dict); throw; }
                g_variant_unref(dict);
                auto next = config;
                const std::vector<std::string> insertion = {"spoken_punctuation", "insert_partials", "stream_insert_delay_ms", "show_overlay"};
                for (auto item = options.begin(); item != options.end(); ++item) {
                    if (!config.contains(item.key()) || item.key() == "backend" || item.key() == "model_profile" || item.key() == "model_family") throw std::invalid_argument("Unknown option: " + item.key());
                    bool is_insertion = std::find(insertion.begin(), insertion.end(), item.key()) != insertion.end();
                    if (is_insertion != (name == "SetInsertionOptions")) throw std::invalid_argument("Option belongs to a different settings group");
                    if (item->type() != config[item.key()].type() && !(item->is_number_integer() && config[item.key()].is_number_integer())) throw std::invalid_argument("Wrong option type: " + item.key());
                    next[item.key()] = *item;
                }
                normalize_paths(next);
                bool restart = false;
                for (auto item = next.begin(); item != next.end(); ++item) {
                    if (*item == config[item.key()]) continue;
                    if (name == "SetRuntimeOptions" && item.key() != "language") { require_idle(); restart = true; }
                    if (item.key() == "insert_partials" && (listening || stopping)) throw std::runtime_error("Stop dictation before changing insertion mode");
                }
                validate(next);
                bool language_changed = next["language"] != config["language"];
                update(next, restart);
                if (worker && language_changed && !listening && !stopping) send({{"command", "set_language"}, {"language", config["language"]}});
            } else if (name == "InstallModel") {
                auto id = string_arg(); require_idle();
                if (id != "nemo-q8" && id != "nemo-q8-english") throw std::invalid_argument("Unknown NeMo model preset");
                auto family = id == "nemo-q8" ? "multilingual" : "english";
                installer = spawn({config["model_installer_path"], "--profile", "nemo-q8", "--model-family", family,
                    "--model-root", config["model_root"], "--streaming-latency-ms", std::to_string(config["streaming_latency_ms"].get<unsigned>())}, Child::Installer, id);
                last_error.clear(); install_progress({{"phase", "starting"}, {"message", "Starting model download"}, {"fraction", 0.0}}, id); changed();
            } else throw std::invalid_argument("Unknown D-Bus method");
            g_dbus_method_invocation_return_value(invocation, nullptr);
        }
    }
};

void Service::read_line(const std::shared_ptr<Child>& child, bool stderr_stream) {
    using Context = std::pair<std::shared_ptr<Child>, bool>;
    g_data_input_stream_read_line_async(stderr_stream ? child->err : child->out, G_PRIORITY_DEFAULT, nullptr,
        [](GObject* stream, GAsyncResult* result, gpointer data) {
            std::unique_ptr<Context> context(static_cast<Context*>(data));
            auto child = context->first; bool err_stream = context->second;
            GError* error = nullptr; gsize length = 0;
            gchar* line = g_data_input_stream_read_line_finish(G_DATA_INPUT_STREAM(stream), result, &length, &error);
            if (!line) {
                if (error) { child->error = error->message; g_error_free(error); }
                (err_stream ? child->stderr_done : child->stdout_done) = true;
                finish_if_ready(child); return;
            }
            std::string text(line, length); g_free(line);
            auto* service = child->owner;
            try {
                if (!service->shutting_down && !text.empty()) {
                    if (child->kind == Child::Worker && service->worker == child) {
                        if (err_stream) std::cerr << "nemote-worker: " << text << '\n';
                        else service->event(Json::parse(text));
                    } else if (child->kind == Child::Installer && service->installer == child) {
                        std::cerr << "nemote-installer: " << text << '\n';
                        if (text.rfind("nemote-progress ", 0) == 0) {
                            auto event = Json::parse(text.substr(std::string("nemote-progress ").size()));
                            if (event.value("phase", "") == "error") child->error = event.value("message", "Model installer failed");
                            service->install_progress(event, child->profile);
                        } else if (err_stream) child->error = text;
                    } else if (child->kind == Child::Devices) {
                        if (err_stream) { child->error = text; std::cerr << "nemote-device: " << text << '\n'; }
                        else { auto event = Json::parse(text); if (event.value("event", "") == "input_device") {
                            auto device = event.at("data");
                            device["selector"] = std::to_string(device.at("index").get<unsigned>());
                            child->devices.push_back(device);
                        } }
                    }
                }
            } catch (const std::exception& error) {
                child->error = error.what();
                child->data_error = true;
                if (child->kind == Child::Worker && service->worker == child) service->fail("Invalid ASR worker event: " + child->error);
            }
            read_line(child, err_stream);
        }, new Context(child, stderr_stream));
}

void Service::finish_if_ready(const std::shared_ptr<Child>& child) {
    if (!child->waited || !child->stdout_done || !child->stderr_done) return;
    if (child->timeout) { g_source_remove(child->timeout); child->timeout = 0; }
    auto* service = child->owner;
    bool success = g_subprocess_get_successful(child->process);
    if (child->kind == Child::Devices && child->invocation) {
        if (success && !child->data_error) g_dbus_method_invocation_return_value(child->invocation, g_variant_new("(@aa{sv})", list_variant(child->devices)));
        else g_dbus_method_invocation_return_dbus_error(child->invocation, "org.freedesktop.DBus.Error.Failed", child->error.empty() ? "Input-device enumeration failed" : child->error.c_str());
    } else if (!service->shutting_down && child->kind == Child::Worker && service->worker == child) {
        service->stop_worker(); service->fail("ASR worker exited unexpectedly");
    } else if (!service->shutting_down && child->kind == Child::Installer && service->installer == child) {
        service->installer.reset();
        std::string message = success ? "Model installed" : (child->error.empty() ? "Model installer exited unexpectedly; retry to resume" : child->error);
        service->install_progress({{"phase", success ? "complete" : "error"}, {"message", message}, {"fraction", success ? 1.0 : 0.0}}, child->profile);
        if (!success) service->fail(message); else { service->last_error.clear(); service->changed(); }
    }
}

int main(int argc, char** argv) {
    try {
        fs::path path = fs::path(g_get_user_config_dir()) / "nemote/service.json";
        bool replace = false;
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--config" && i + 1 < argc) path = argv[++i];
            else if (arg == "--replace") replace = true;
            else if (arg == "--help") { std::cout << "nemote-service [--config PATH] [--replace]\n"; return 0; }
            else throw std::invalid_argument("Unknown argument: " + arg);
        }
        Service service(fs::absolute(path)); service.loop = g_main_loop_new(nullptr, false);
        GError* error = nullptr;
        service.bus = g_bus_get_sync(G_BUS_TYPE_SESSION, nullptr, &error);
        if (!service.bus) { std::string message = error->message; g_error_free(error); throw std::runtime_error(message); }
        auto* info = g_dbus_node_info_new_for_xml(service_xml, &error);
        if (!info) { std::string message = error->message; g_error_free(error); throw std::runtime_error(message); }
        static const GDBusInterfaceVTable vtable = {
            [](GDBusConnection*, const gchar* sender, const gchar*, const gchar*, const gchar* method, GVariant* params, GDBusMethodInvocation* invocation, gpointer data) {
                auto* service = static_cast<Service*>(data);
                try { service->method(method, params, sender, invocation); }
                catch (const std::invalid_argument& error) { g_dbus_method_invocation_return_dbus_error(invocation, "org.freedesktop.DBus.Error.InvalidArgs", error.what()); }
                catch (const std::exception& error) { service->fail(error.what()); g_dbus_method_invocation_return_dbus_error(invocation, "org.freedesktop.DBus.Error.Failed", error.what()); }
            }, nullptr, nullptr, {nullptr}
        };
        auto registration = g_dbus_connection_register_object(service.bus, object_path, info->interfaces[0], &vtable, &service, nullptr, &error);
        if (!registration) { std::string message = error->message; g_error_free(error); throw std::runtime_error(message); }
        auto owner = g_bus_own_name_on_connection(service.bus, bus_name,
            GBusNameOwnerFlags(G_BUS_NAME_OWNER_FLAGS_ALLOW_REPLACEMENT | (replace ? G_BUS_NAME_OWNER_FLAGS_REPLACE : 0)),
            [](GDBusConnection*, const gchar*, gpointer) { std::cerr << "nemote-service: listening on " << bus_name << " " << object_path << '\n'; },
            [](GDBusConnection*, const gchar*, gpointer data) { static_cast<Service*>(data)->shutdown(); }, &service, nullptr);
        auto subscription = g_dbus_connection_signal_subscribe(service.bus, "org.freedesktop.DBus", "org.freedesktop.DBus", "NameOwnerChanged", "/org/freedesktop/DBus", nullptr, G_DBUS_SIGNAL_FLAGS_NONE,
            [](GDBusConnection*, const gchar*, const gchar*, const gchar*, const gchar*, GVariant* params, gpointer data) {
                const gchar *name, *old_owner, *new_owner; g_variant_get(params, "(&s&s&s)", &name, &old_owner, &new_owner);
                auto* service = static_cast<Service*>(data);
                if (!service->shell.empty() && service->shell == name && !*new_owner) service->shutdown();
            }, &service, nullptr);
        auto term = g_unix_signal_add(SIGTERM, [](gpointer data) -> gboolean { static_cast<Service*>(data)->shutdown(); return G_SOURCE_CONTINUE; }, &service);
        auto interrupt = g_unix_signal_add(SIGINT, [](gpointer data) -> gboolean { static_cast<Service*>(data)->shutdown(); return G_SOURCE_CONTINUE; }, &service);
        g_main_loop_run(service.loop);
        service.shutdown_children();
        g_source_remove(term); g_source_remove(interrupt);
        g_dbus_connection_signal_unsubscribe(service.bus, subscription);
        g_bus_unown_name(owner); g_dbus_connection_unregister_object(service.bus, registration);
        g_dbus_node_info_unref(info);
        g_dbus_connection_flush_sync(service.bus, nullptr, nullptr);
        // Drain the async subprocess callbacks before destroying their owner.
        while (g_main_context_pending(nullptr)) g_main_context_iteration(nullptr, false);
        g_object_unref(service.bus); service.bus = nullptr;
        g_main_loop_unref(service.loop);
        return 0;
    } catch (const std::exception& error) { std::cerr << "nemote-service: " << error.what() << '\n'; return 1; }
}
