// Wordpipe's native NeMo backend. The process protocol is the backend interface;
// audio, frontend, caches, decoder and EOF processing stay entirely in C++.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "recognizer.h"
#include "transcript.h"
#include "session_thread.h"
#include <ggml-backend.h>
#include "nlohmann/json.hpp"
#define MA_NO_ENCODING
#define MA_NO_RESOURCE_MANAGER
#define MA_NO_NODE_GRAPH
#define MA_NO_ENGINE
#define MA_NO_GENERATION
#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"

using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;
namespace asr = nemo_speech::asr;
static std::mutex output_mutex;
static void emit(const Json& event) {
    std::lock_guard<std::mutex> guard(output_mutex);
    std::cout << event.dump() << std::endl;
}
static double seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

struct Args {
    std::string model_dir, language = "en-US", input_device, wav;
    std::string device = "auto";
    int threads = 2, sample_rate = 16000, chunk_samples = 8960;
    int wav_repeat = 1;
    double queue_seconds = 10, stats_interval = 1;
    bool list_devices = false;
};
static Args parse_args(int argc, char** argv) {
    Args args;
    if (const char* device = std::getenv("WORDPIPE_NEMO_DEVICE")) args.device = device;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        if (key == "--list-input-devices") { args.list_devices = true; continue; }
        if (key == "--help") {
            std::cout << "wordpipe-nemo-worker --model-dir DIR|GGUF [--wav WAV] "
                         "[--chunk-samples 1280|2560|8960|17920] [--num-threads N] "
                         "[--language en-US|auto] [--input-device NAME|INDEX] "
                         "[--device auto|cpu]\n";
            std::exit(0);
        }
        if (i + 1 == argc) throw std::invalid_argument("missing value for " + key);
        std::string value = argv[++i];
        if (key == "--model-dir") args.model_dir = value;
        else if (key == "--language") args.language = value;
        else if (key == "--input-device") args.input_device = value;
        else if (key == "--wav") args.wav = value;
        else if (key == "--wav-repeat") args.wav_repeat = std::stoi(value);
        else if (key == "--device") args.device = value;
        else if (key == "--num-threads") args.threads = std::stoi(value);
        else if (key == "--sample-rate") args.sample_rate = std::stoi(value);
        else if (key == "--chunk-samples") args.chunk_samples = std::stoi(value);
        else if (key == "--queue-seconds") args.queue_seconds = std::stod(value);
        else if (key == "--stats-interval-seconds") args.stats_interval = std::stod(value);
        else throw std::invalid_argument("unknown option: " + key);
    }
    if (!args.list_devices && args.model_dir.empty()) throw std::invalid_argument("--model-dir is required");
    if (args.sample_rate != 16000) throw std::invalid_argument("--sample-rate must be 16000");
    if (args.device != "auto" && args.device != "cpu")
        throw std::invalid_argument("--device must be auto or cpu; GPU acceleration is not enabled in this worker");
    if (args.threads < 1) throw std::invalid_argument("--num-threads must be positive");
    if (args.wav_repeat < 1 || args.wav_repeat > 100) throw std::invalid_argument("--wav-repeat must be in [1,100]");
    if (args.chunk_samples != 1280 && args.chunk_samples != 2560 &&
        args.chunk_samples != 8960 && args.chunk_samples != 17920)
        throw std::invalid_argument("chunk size must be 80, 160, 560, or 1120 ms");
    if (!std::isfinite(args.queue_seconds) || args.queue_seconds <= 0 || args.queue_seconds > 60 ||
        !std::isfinite(args.stats_interval) || args.stats_interval <= 0)
        throw std::invalid_argument("queue seconds must be in (0,60]; stats interval must be positive");
    return args;
}

static void initialize_backends() {
#ifdef WORDPIPE_NEMO_DYNAMIC_BACKENDS
    // Relocated release libraries take precedence; the matching SDK path is
    // only a development-build fallback. Load modules before device discovery.
    auto executable = std::filesystem::canonical("/proc/self/exe");
    auto libraries = executable.parent_path().parent_path() / "lib/nemo";
    if (!std::filesystem::is_directory(libraries)) libraries = WORDPIPE_NEMO_BUILD_LIBRARY_DIR;
    ggml_backend_load_all_from_path(libraries.c_str());
#endif
}

static Json cpu_features() {
    Json features = Json::object();
    auto cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!cpu) return features;
    auto reg = ggml_backend_dev_backend_reg(cpu);
    auto get_features = reinterpret_cast<ggml_backend_get_features_t>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_get_features"));
    if (get_features)
        for (auto feature = get_features(reg); feature && feature->name; ++feature)
            features[feature->name] = feature->value ? feature->value : "";
    return features;
}

class Capture {
    ma_context context_{};
    ma_device device_{};
    bool initialized_ = false;
    std::vector<float> ring_;
    std::atomic<size_t> read_{0}, write_{0}, dropped_{0};
    std::string name_;
    static void callback(ma_device* device, void*, const void* input, ma_uint32 frames) {
        if (!input) return;
        auto& self = *static_cast<Capture*>(device->pUserData);
        auto write = self.write_.load(std::memory_order_relaxed);
        auto read = self.read_.load(std::memory_order_acquire);
        size_t count = std::min<size_t>(frames, self.ring_.size() - (write - read));
        auto offset = write % self.ring_.size();
        auto first = std::min(count, self.ring_.size() - offset);
        auto samples = static_cast<const float*>(input);
        std::memcpy(self.ring_.data() + offset, samples, first * sizeof(float));
        std::memcpy(self.ring_.data(), samples + first, (count - first) * sizeof(float));
        if (count < frames) self.dropped_.fetch_add(1, std::memory_order_relaxed);
        self.write_.store(write + count, std::memory_order_release);
    }
public:
    explicit Capture(const Args& args) : ring_(std::max<size_t>(1, static_cast<size_t>(args.queue_seconds * 16000))) {
        if (ma_context_init(nullptr, 0, nullptr, &context_) != MA_SUCCESS)
            throw std::runtime_error("could not initialize audio capture");
    }
    ~Capture() { stop(); ma_context_uninit(&context_); }
    void enumerate() {
        ma_device_info *playback, *capture;
        ma_uint32 nplayback, ncapture;
        if (ma_context_get_devices(&context_, &playback, &nplayback, &capture, &ncapture) != MA_SUCCESS)
            throw std::runtime_error("could not enumerate input devices");
        for (ma_uint32 i = 0; i < ncapture; ++i)
            emit({{"event", "input_device"}, {"data", {{"index", i}, {"name", capture[i].name}, {"is_default", bool(capture[i].isDefault)}}}});
    }
    void start(const Args& args) {
        ma_device_config config = ma_device_config_init(ma_device_type_capture);
        config.capture.format = ma_format_f32;
        config.capture.channels = 1;
        config.sampleRate = 16000;
        config.periodSizeInMilliseconds = 40;
        config.dataCallback = callback;
        config.pUserData = this;
        ma_device_id selected{};
        if (!args.input_device.empty()) {
            ma_device_info *playback, *capture;
            ma_uint32 nplayback, ncapture;
            if (ma_context_get_devices(&context_, &playback, &nplayback, &capture, &ncapture) != MA_SUCCESS)
                throw std::runtime_error("could not enumerate input devices");
            bool found = false;
            for (ma_uint32 i = 0; i < ncapture; ++i) {
                if (args.input_device == std::to_string(i) || std::string(capture[i].name).find(args.input_device) != std::string::npos) {
                    selected = capture[i].id; found = true; break;
                }
            }
            if (!found) throw std::invalid_argument("input device not found: " + args.input_device);
            config.capture.pDeviceID = &selected;
        }
        if (ma_device_init(&context_, &config, &device_) != MA_SUCCESS)
            throw std::runtime_error("could not open microphone");
        initialized_ = true;
        name_ = device_.capture.name;
        if (ma_device_start(&device_) != MA_SUCCESS) throw std::runtime_error("could not start microphone");
    }
    void stop() { if (initialized_) { ma_device_uninit(&device_); initialized_ = false; } }
    std::vector<float> drain() {
        auto read = read_.load(std::memory_order_relaxed);
        auto count = write_.load(std::memory_order_acquire) - read;
        std::vector<float> samples(count);
        for (size_t i = 0; i < count; ++i) samples[i] = ring_[(read + i) % ring_.size()];
        read_.store(read + count, std::memory_order_release);
        return samples;
    }
    size_t dropped() const { return dropped_.load(); }
    const std::string& name() const { return name_; }
};

// Complete snapshots (not token deltas) match the existing worker protocol.
// Endpoint finals, interim rewrites, and late punctuation are owned by upstream.
class Session {
    std::unique_ptr<asr::RecognitionStream> stream_;
    const Args& args_;
    Clock::time_point started_ = Clock::now();
    Transcript transcript_;
    size_t accepted_ = 0, calls_ = 0, dropped_ = 0;
    double decode_seconds_ = 0, rms_ = 0, peak_ = 0;
    void update(const asr::Result& result) {
        std::string text = result.alternatives.empty() ? "" : result.alternatives.front().transcript;
        transcript_.update(text, result.is_final, result.late_punctuation);
    }
public:
    Session(asr::Recognizer& model, const Args& args) : args_(args) {
        auto language = args.language == "auto" ? "" : args.language;
        auto supported = model.supported_languages();
        if (!language.empty() && !supported.empty() &&
            std::find(supported.begin(), supported.end(), language) == supported.end()) {
            auto base = language.substr(0, language.find('-'));
            auto match = std::find_if(supported.begin(), supported.end(), [&](const auto& entry) {
                return entry.substr(0, entry.find('-')) == base;
            });
            if (match == supported.end()) throw std::invalid_argument("language is not supported by this model: " + args.language);
            language = *match;
        }
        asr::AsrRequestOptions options;
        options.enable_automatic_punctuation = true; // Preserve model-native formatting in finals too.
        stream_ = model.streaming_recognize(options, language, false);
    }
    Json metrics() const {
        auto audio = double(accepted_) / 16000;
        return {{"audio_seconds", audio}, {"processed_audio_seconds", audio},
            {"synthetic_audio_seconds", 0}, {"elapsed_seconds", seconds(started_)},
            {"decode_seconds", decode_seconds_}, {"decode_calls", calls_},
            {"real_time_factor", audio > 0 ? decode_seconds_ / audio : 0},
            {"real_audio_real_time_factor", audio > 0 ? decode_seconds_ / audio : 0},
            {"dropped_audio_chunks", dropped_}, {"last_rms", rms_}, {"peak_rms", peak_}};
    }
    void stats() { emit({{"event", "stats"}, {"text", transcript_.text()}, {"data", metrics()}}); }
    void push(const float* audio, size_t n, size_t dropped = 0) {
        if (!n) return;
        double power = 0;
        for (size_t i = 0; i < n; ++i) { power += audio[i] * audio[i]; peak_ = std::max(peak_, double(std::abs(audio[i]))); }
        rms_ = std::sqrt(power / n);
        accepted_ += n; dropped_ = dropped;
        auto start = Clock::now();
        stream_->push(audio, n, 16000);
        auto previous = transcript_.text();
        while (auto result = stream_->next()) update(*result);
        decode_seconds_ += seconds(start); ++calls_;
        if (transcript_.text() != previous && !transcript_.text().empty())
            emit({{"event", "partial"}, {"text", transcript_.text()}, {"data", metrics()}});
    }
    void finish() {
        auto start = Clock::now();
        update(stream_->finish());
        decode_seconds_ += seconds(start); ++calls_;
        if (!transcript_.text().empty()) emit({{"event", "commit"}, {"text", transcript_.text()}, {"data", metrics()}});
    }
};

static void run_capture(asr::Recognizer& model, const Args& args, std::atomic<bool>& stop) {
    try {
        Session session(model, args);
        Capture capture(args);
        capture.start(args);
        emit({{"event", "listening"}, {"data", {{"input_device", {{"name", capture.name()}, {"requested", args.input_device}}}}}});
        auto stats = Clock::now();
        while (!stop.load()) {
            auto samples = capture.drain();
            session.push(samples.data(), samples.size(), capture.dropped());
            if (seconds(stats) >= args.stats_interval) { session.stats(); stats = Clock::now(); }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        capture.stop();
        auto samples = capture.drain();
        session.push(samples.data(), samples.size(), capture.dropped());
        session.finish();
    } catch (const std::exception& error) {
        emit({{"event", "error"}, {"message", error.what()}});
    }
}

int main(int argc, char** argv) {
    try {
        Args args = parse_args(argc, argv);
        if (args.list_devices) { Capture(args).enumerate(); return 0; }
        auto path = std::filesystem::path(args.model_dir);
        if (std::filesystem::is_directory(path)) path /= "model.gguf";
        if (!std::filesystem::is_regular_file(path)) throw std::invalid_argument("GGUF model is not installed at " + path.string());
        initialize_backends();
        asr::RecognizerConfig config;
        config.backend.gpu = -1; config.backend.threads = args.threads;
        config.model.path = path.string();
        config.streaming.rnnt_right_context = args.chunk_samples / 1280 - 1;
        config.batching.enabled = false; config.log_status = false;
        emit({{"event", "loading_model"}, {"data", {{"model_dir", args.model_dir}}}});
        auto start = Clock::now();
        asr::Recognizer model(config);
        if (model.sample_rate() != 16000) throw std::invalid_argument("this worker requires a 16 kHz model");
        { Session validate(model, args); }
        emit({{"event", "model_loaded"}, {"data", {{"load_seconds", seconds(start)}, {"num_threads", args.threads},
             {"compute_backend", "cpu"}, {"cpu_features", cpu_features()}}}});
        if (!args.wav.empty()) {
            for (int repeat = 0; repeat < args.wav_repeat; ++repeat) {
            ma_decoder_config decode_config = ma_decoder_config_init(ma_format_f32, 1, 16000);
            ma_decoder decoder;
            if (ma_decoder_init_file(args.wav.c_str(), &decode_config, &decoder) != MA_SUCCESS)
                throw std::runtime_error("could not read WAV: " + args.wav);
            struct Cleanup { ma_decoder* p; ~Cleanup() { ma_decoder_uninit(p); } } cleanup{&decoder};
            Session session(model, args);
            std::vector<float> audio(args.chunk_samples);
            ma_uint64 count = 0;
            while (true) {
                auto status = ma_decoder_read_pcm_frames(&decoder, audio.data(), audio.size(), &count);
                session.push(audio.data(), count);
                if (status == MA_AT_END || count == 0) break;
                if (status != MA_SUCCESS) throw std::runtime_error("WAV decode failed");
            }
            session.finish();
            }
            return 0;
        }
        emit({{"event", "ready"}});
        SessionThread session;
        // Also stop/join on malformed commands, shutdown, or stdin EOF.
        auto join = [&] { session.stop(); };
        std::string line;
        while (std::getline(std::cin, line)) {
            if (line.empty()) continue;
            try {
                auto command = Json::parse(line);
                auto name = command.at("command").get<std::string>();
                if (name == "start") {
                    session.start([&](std::atomic<bool>& stop) { run_capture(model, args, stop); },
                        [] { emit({{"event", "stopped"}}); });
                } else if (name == "stop") { if (session.active()) join(); else { join(); emit({{"event", "stopped"}}); } }
                else if (name == "shutdown") { join(); break; }
                else if (name == "set_language") {
                    if (session.active()) throw std::invalid_argument("cannot change language while recognition is active");
                    session.stop();
                    Args next = args; next.language = command.at("language").get<std::string>();
                    Session validate(model, next);
                    args.language = next.language;
                    emit({{"event", "language_changed"}, {"data", {{"language", args.language}}}});
                } else throw std::invalid_argument("unknown command: " + name);
            } catch (const std::exception& error) { emit({{"event", "error"}, {"message", error.what()}}); }
        }
        join();
    } catch (const std::exception& error) {
        emit({{"event", "error"}, {"message", error.what()}});
        std::cerr << "wordpipe-nemo-worker: " << error.what() << '\n';
        return 1;
    }
}
