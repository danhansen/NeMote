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
#include "itn_format.h"
#include <sstream>
#include <spawn.h>
#include <sys/wait.h>
#include <cerrno>
extern char** environ;
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
    bool itn = false;
    bool phrase_boosting = false, vad_filtering = false;
    std::string boost_phrases, vad_model;
    std::string boost_tokenizer;
    std::string model_family, companion_installer;
    std::string endpoint_mode = "disabled";
    std::string itn_grammar_dir;
};
static Args parse_args(int argc, char** argv) {
    Args args;
    if (const char* device = std::getenv("WORDPIPE_NEMO_DEVICE")) args.device = device;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        if (key == "--list-input-devices") { args.list_devices = true; continue; }
        if (key == "--itn") { args.itn = true; continue; }
        if (key == "--phrase-boosting") { args.phrase_boosting = true; continue; }
        if (key == "--vad-filtering") { args.vad_filtering = true; continue; }
        if (key == "--help") {
            std::cout << "wordpipe-nemo-worker --model-dir DIR|GGUF [--wav WAV] "
                         "[--chunk-samples 1280|2560|8960|17920] [--num-threads N] "
                         "[--language en-US|auto] [--input-device NAME|INDEX] "
                         "[--device auto|cpu] [--itn] [--phrase-boosting --boost-phrases TEXT [--boost-tokenizer MODEL]] "
                         "[--vad-filtering [--vad-model GGUF]] [--model-family english|multilingual] "
                         "[--companion-installer PATH]\n";
            std::exit(0);
        }
        if (i + 1 == argc) throw std::invalid_argument("missing value for " + key);
        std::string value = argv[++i];
        if (key == "--model-dir") args.model_dir = value;
        else if (key == "--language") args.language = value;
        else if (key == "--boost-phrases") args.boost_phrases = value;
        else if (key == "--boost-tokenizer") args.boost_tokenizer = value;
        else if (key == "--model-family") args.model_family = value;
        else if (key == "--companion-installer") args.companion_installer = value;
        else if (key == "--endpoint-mode") args.endpoint_mode = value;
        else if (key == "--vad-model") args.vad_model = value;
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
    if (args.endpoint_mode != "disabled" && args.endpoint_mode != "preview" && args.endpoint_mode != "commit")
        throw std::invalid_argument("--endpoint-mode must be disabled, preview, or commit");
    if (args.phrase_boosting && args.boost_phrases.find_first_not_of(" \t\r\n") == std::string::npos)
        throw std::invalid_argument("Phrase boosting requires at least one phrase");
    if (args.boost_phrases.size() > 16384) throw std::invalid_argument("Boost phrases exceed 16 KiB");
    if (args.vad_filtering && !args.vad_model.empty() && !std::filesystem::is_regular_file(args.vad_model))
        throw std::invalid_argument("VAD filtering requires an existing Silero VAD GGUF (--vad-model)");
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
    asr::Recognizer& model_;
    Clock::time_point started_ = Clock::now();
    Transcript transcript_;
    size_t accepted_ = 0, calls_ = 0, dropped_ = 0;
    double decode_seconds_ = 0, rms_ = 0, peak_ = 0;
    bool english_model_ = false;
    std::string preview_raw_, preview_formatted_;
    size_t preview_endpoints_ = 0, utterance_commits_ = 0;
    void update(const asr::Result& result) {
        std::string text = result.alternatives.empty() ? "" : result.alternatives.front().transcript;
        auto language = args_.language;
        if (language == "auto" && !result.alternatives.empty()) {
            const auto& codes = result.alternatives.front().language_codes;
            language = codes.empty() ? (english_model_ ? "en" : "") : codes.front();
        }
        if (result.pause_endpoint) {
            ++preview_endpoints_;
            preview_raw_ = text;
            preview_formatted_ = args_.itn ? model_.postproc().apply(text, stream_->options(), nullptr, language) : text;
            if (args_.itn && language.substr(0, language.find('-')) == "en")
                preview_formatted_ = format_english_itn(preview_formatted_);
        }
        if (result.is_final) {
            preview_raw_.clear(); preview_formatted_.clear();
        } else if (!preview_raw_.empty() && text == preview_raw_) {
            text = preview_formatted_;
        } else {
            preview_raw_.clear(); preview_formatted_.clear();
        }
        if (args_.itn && result.is_final && !result.alternatives.empty()) {
            if (language.substr(0, language.find('-')) == "en") text = format_english_itn(text);
        }
        transcript_.update(text, result.is_final, result.late_punctuation);
    }
public:
    Session(asr::Recognizer& model, const Args& args) : args_(args), model_(model) {
        if (args.itn && args.language != "auto") {
            auto base = args.language.substr(0, args.language.find('-'));
            if (!std::filesystem::is_regular_file(std::filesystem::path(args.itn_grammar_dir) / base / "verbalize.far"))
                throw std::invalid_argument("ITN grammars are unavailable for language " + args.language);
        }
        auto language = args.language == "auto" ? "" : args.language;
        auto supported = model.supported_languages();
        english_model_ = supported.empty() || std::all_of(supported.begin(), supported.end(), [](const auto& code) {
            return code.substr(0, code.find('-')) == "en";
        });
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
        options.verbatim_transcripts = !args.itn;
        if (args.phrase_boosting) {
            asr::AsrRequestOptions::Boost context;
            context.boost = 1.0f; // Upstream's nominal RNNT shallow-fusion weight.
            std::istringstream lines(args.boost_phrases);
            std::string phrase;
            while (std::getline(lines, phrase)) {
                const auto first = phrase.find_first_not_of(" \t\r");
                if (first != std::string::npos)
                    context.phrases.push_back(phrase.substr(first, phrase.find_last_not_of(" \t\r") - first + 1));
            }
            options.speech_contexts.push_back(std::move(context));
        }
        options.enable_automatic_punctuation = true; // Preserve model-native formatting in finals too.
        stream_ = model.streaming_recognize(options, language, false);
        if (args.endpoint_mode == "preview") stream_->set_interim_words(true); // Detected language, not word offsets.
    }
    Json metrics() const {
        auto audio = double(accepted_) / 16000;
        return {{"audio_seconds", audio}, {"processed_audio_seconds", audio},
            {"synthetic_audio_seconds", 0}, {"elapsed_seconds", seconds(started_)},
            {"decode_seconds", decode_seconds_}, {"decode_calls", calls_},
            {"real_time_factor", audio > 0 ? decode_seconds_ / audio : 0},
            {"real_audio_real_time_factor", audio > 0 ? decode_seconds_ / audio : 0},
            {"dropped_audio_chunks", dropped_}, {"last_rms", rms_}, {"peak_rms", peak_},
            {"preview_endpoints", preview_endpoints_}, {"utterance_commits", utterance_commits_}};
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
        std::vector<std::string> commits;
        while (auto result = stream_->next()) {
            update(*result);
            if (result->is_final && args_.endpoint_mode == "commit" && !transcript_.text().empty()) {
                ++utterance_commits_;
                commits.push_back(transcript_.text());
            }
        }
        decode_seconds_ += seconds(start); ++calls_;
        for (const auto& text : commits)
            emit({{"event", "commit"}, {"text", text}, {"data", metrics()}});
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

static void prepare_companions(Args& args, const std::filesystem::path& model) {
    const auto directory = model.parent_path();
    bool boosting = args.phrase_boosting && args.boost_tokenizer.empty() &&
                    !std::filesystem::is_regular_file(directory / "tokenizer.model");
    if (args.vad_filtering && args.vad_model.empty())
        args.vad_model = (directory / "silero-v6.2.0.gguf").string();
    const bool vad = args.vad_filtering && !std::filesystem::is_regular_file(args.vad_model);
    if (!boosting && !vad) return;
    if (args.model_family != "english" && args.model_family != "multilingual")
        throw std::invalid_argument("Automatic companion downloads require --model-family english|multilingual");
    auto installer = args.companion_installer.empty()
        ? std::filesystem::canonical("/proc/self/exe").parent_path() / "wordpipe-companion-install"
        : std::filesystem::path(args.companion_installer);
    if (!std::filesystem::is_regular_file(installer))
        throw std::runtime_error("Companion download helper is missing: " + installer.string());
    std::vector<std::string> command{installer.string(), "--runtime-dir", directory.string(),
                                     "--model-family", args.model_family};
    if (boosting) command.push_back("--boosting");
    if (vad) command.push_back("--vad");
    std::vector<char*> argv;
    for (auto& item : command) argv.push_back(item.data());
    argv.push_back(nullptr);
    pid_t pid;
    const int error = posix_spawn(&pid, command.front().c_str(), nullptr, nullptr, argv.data(), environ);
    if (error) throw std::runtime_error("Could not launch companion installer: " + std::string(std::strerror(error)));
    int status = 0;
    pid_t result;
    do { result = waitpid(pid, &status, 0); } while (result == -1 && errno == EINTR);
    if (result == -1 || !WIFEXITED(status) || WEXITSTATUS(status) != 0)
        throw std::runtime_error("NeMo companion download failed; see service logs for details");
    if (boosting && !std::filesystem::is_regular_file(directory / "tokenizer.model"))
        throw std::runtime_error("Companion installer did not produce a tokenizer");
    if (vad && !std::filesystem::is_regular_file(args.vad_model))
        throw std::runtime_error("Companion installer did not produce the VAD model");
}

int main(int argc, char** argv) {
    try {
        Args args = parse_args(argc, argv);
        if (args.list_devices) { Capture(args).enumerate(); return 0; }
        auto path = std::filesystem::path(args.model_dir);
        if (std::filesystem::is_directory(path)) path /= "model.gguf";
        if (!std::filesystem::is_regular_file(path)) throw std::invalid_argument("GGUF model is not installed at " + path.string());
        emit({{"event", "loading_model"}, {"data", {{"model_dir", args.model_dir}}}});
        prepare_companions(args, path);
        initialize_backends();
        asr::RecognizerConfig config;
        config.backend.gpu = -1; config.backend.threads = args.threads;
        config.model.path = path.string();
        if (args.phrase_boosting) {
            auto tokenizer = args.boost_tokenizer.empty() ? path.parent_path() / "tokenizer.model"
                                                        : std::filesystem::path(args.boost_tokenizer);
            if (std::filesystem::is_regular_file(tokenizer)) config.decoder.tokenizer_path = tokenizer.string();
            else if (!args.boost_tokenizer.empty()) throw std::invalid_argument("Boosting tokenizer file does not exist");
        }
        if (args.vad_filtering) {
            config.vad.model_path = args.vad_model;
            config.vad.masker.mask_enable = true;
        }
        if (args.itn) {
#ifdef WORDPIPE_NEMO_ITN
            auto executable = std::filesystem::canonical("/proc/self/exe");
            auto grammars = executable.parent_path().parent_path() / "share/wordpipe/itn";
            if (!std::filesystem::is_directory(grammars)) grammars = WORDPIPE_NEMO_ITN_GRAMMAR_DIR;
            auto language = args.language.substr(0, args.language.find('-'));
            if (language != "auto" && !std::filesystem::is_regular_file(grammars / language / "verbalize.far"))
                throw std::invalid_argument("ITN grammars are unavailable for language " + args.language);
            config.postproc.itn_model_dir = grammars.string();
            args.itn_grammar_dir = grammars.string();
#else
            throw std::invalid_argument("This worker was built without ITN support");
#endif
        }
        config.streaming.rnnt_right_context = args.chunk_samples / 1280 - 1;
        config.endpointing.enable = args.endpoint_mode != "disabled";
        config.endpointing.preview_only = args.endpoint_mode == "preview";
        config.batching.enabled = false; config.log_status = false;
        auto start = Clock::now();
        asr::Recognizer model(config);
        if (model.sample_rate() != 16000) throw std::invalid_argument("this worker requires a 16 kHz model");
        { Session validate(model, args); }
        emit({{"event", "model_loaded"}, {"data", {{"load_seconds", seconds(start)}, {"num_threads", args.threads},
             {"compute_backend", "cpu"}, {"cpu_features", cpu_features()}, {"itn", args.itn},
             {"phrase_boosting", args.phrase_boosting}, {"vad_filtering", args.vad_filtering},
             {"endpoint_mode", args.endpoint_mode}}}});
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
