// sa3-server: HTTP server for Stable Audio 3 via sa3.cpp's libsa3 C ABI (CUDA).
// Speaks ace-server's async job protocol (/health, /props, /lm, /synth, /job)
// with constant-time Bearer auth from SA3_API_KEY or --api-key-file.
// Loads weights only while a song renders (frugal stage-by-stage offloading) and
// releases the backend immediately after, holding no GPU memory between songs.

#include "libsa3_v1.h"

#include "httplib.h"
#include "yyjson.h"

#include <lame/lame.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr int    MAX_SECONDS       = 180;
constexpr int    DEFAULT_STEPS     = 8;
constexpr int    DEFAULT_BITRATE   = 192;
constexpr size_t MAX_BODY_BYTES    = 1u << 20;
constexpr size_t MAX_PROMPT_CHARS  = 4096;
constexpr size_t MAX_PENDING_JOBS  = 8;
constexpr size_t MAX_FINISHED_JOBS = 8;
constexpr auto   JOB_TTL           = std::chrono::seconds(600);
constexpr char   BOUNDARY[]        = "sa3-batch-boundary";

struct DiscoveredModels {
    std::vector<std::string> dit;
    std::vector<std::string> decoder;
    std::vector<std::string> text_encoder;
    std::vector<std::string> tokenizer;
    std::vector<std::string> conditioner;
};

static bool ends_with(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() &&
           s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

static DiscoveredModels discover_models(const fs::path& models_dir) {
    DiscoveredModels found;
    std::error_code ec;
    if (!fs::is_directory(models_dir, ec)) return found;

    std::vector<std::string> rels;
    for (const auto& entry : fs::recursive_directory_iterator(models_dir, ec)) {
        if (ec) break;
        if (!entry.is_regular_file(ec)) continue;
        auto rel = fs::relative(entry.path(), models_dir, ec);
        if (!ec) rels.push_back(rel.generic_string());
    }
    std::sort(rels.begin(), rels.end());

    for (const auto& rel : rels) {
        std::string name = fs::path(rel).filename().string();
        if (!ends_with(name, ".gguf")) continue;
        if (name.find("-dit-") != std::string::npos) {
            found.dit.push_back(rel);
        } else if (name.find("-same-") != std::string::npos) {
            found.decoder.push_back(rel);
        } else if (name.find("encoder") != std::string::npos) {
            found.text_encoder.push_back(rel);
        } else if (name.find("vocab") != std::string::npos) {
            found.tokenizer.push_back(rel);
        } else if (name.find("-conditioner-") != std::string::npos) {
            found.conditioner.push_back(rel);
        }
    }
    return found;
}

static bool constant_time_equal(const std::string& a, const std::string& b) {
    const size_t max_len = std::max(a.size(), b.size());
    unsigned char diff = static_cast<unsigned char>(a.size() != b.size());
    for (size_t i = 0; i < max_len; ++i) {
        const unsigned char ca = i < a.size() ? static_cast<unsigned char>(a[i]) : 0;
        const unsigned char cb = i < b.size() ? static_cast<unsigned char>(b[i]) : 0;
        diff |= static_cast<unsigned char>(ca ^ cb);
    }
    return diff == 0;
}

static std::string random_hex_id() {
    static thread_local std::mt19937_64 rng(std::random_device{}());
    uint64_t val = rng();
    std::ostringstream oss;
    oss << std::hex << std::setfill('0') << std::setw(16) << val;
    return oss.str();
}

static int64_t random_seed() {
    static thread_local std::mt19937 rng(std::random_device{}());
    return static_cast<int64_t>(rng() & 0x7fffffffu);
}

static std::string encode_mp3(const float* samples, uint64_t n_samples,
                              uint32_t n_channels, uint32_t sample_rate,
                              sa3_audio_layout_v1 layout, int bitrate_kbps) {
    if (!samples || n_samples == 0 || n_channels == 0 || n_channels > 2) {
        throw std::runtime_error("invalid audio buffer for MP3 encoding");
    }

    // Clamp to [-1, 1] and prepare planar stereo float buffers for LAME.
    std::vector<float> left(n_samples);
    std::vector<float> right(n_samples);
    if (n_channels == 2) {
        if (layout == SA3_AUDIO_PLANAR_V1) {
            const float* src_l = samples;
            const float* src_r = samples + n_samples;
            for (uint64_t i = 0; i < n_samples; ++i) {
                left[i]  = std::clamp(src_l[i], -1.0f, 1.0f);
                right[i] = std::clamp(src_r[i], -1.0f, 1.0f);
            }
        } else {
            for (uint64_t i = 0; i < n_samples; ++i) {
                left[i]  = std::clamp(samples[2 * i],     -1.0f, 1.0f);
                right[i] = std::clamp(samples[2 * i + 1], -1.0f, 1.0f);
            }
        }
    } else {
        for (uint64_t i = 0; i < n_samples; ++i) {
            float v = std::clamp(samples[i], -1.0f, 1.0f);
            left[i]  = v;
            right[i] = v;
        }
    }

    lame_t gfp = lame_init();
    if (!gfp) throw std::runtime_error("lame_init failed");
    struct LameGuard {
        lame_t g;
        ~LameGuard() { if (g) lame_close(g); }
    } guard{gfp};

    lame_set_in_samplerate(gfp, static_cast<int>(sample_rate));
    lame_set_num_channels(gfp, 2);
    lame_set_mode(gfp, JOINT_STEREO);
    lame_set_brate(gfp, bitrate_kbps);
    lame_set_quality(gfp, 2);
    lame_set_bWriteVbrTag(gfp, 0);
    if (lame_init_params(gfp) < 0) {
        throw std::runtime_error("lame_init_params failed");
    }

    std::string out;
    out.reserve(static_cast<size_t>((n_samples / std::max<uint32_t>(sample_rate, 1)) * (bitrate_kbps * 128) + 32768));

    constexpr uint64_t CHUNK = 65536;
    std::vector<unsigned char> buf(static_cast<size_t>(CHUNK * 5 / 4 + 7200));

    uint64_t offset = 0;
    while (offset < n_samples) {
        const int n = static_cast<int>(std::min<uint64_t>(CHUNK, n_samples - offset));
        const int written = lame_encode_buffer_ieee_float(
            gfp, left.data() + offset, right.data() + offset, n,
            buf.data(), static_cast<int>(buf.size()));
        if (written < 0) {
            throw std::runtime_error("lame_encode_buffer_ieee_float failed (" + std::to_string(written) + ")");
        }
        if (written > 0) out.append(reinterpret_cast<const char*>(buf.data()), static_cast<size_t>(written));
        offset += static_cast<uint64_t>(n);
    }

    const int flushed = lame_encode_flush(gfp, buf.data(), static_cast<int>(buf.size()));
    if (flushed < 0) {
        throw std::runtime_error("lame_encode_flush failed (" + std::to_string(flushed) + ")");
    }
    if (flushed > 0) out.append(reinterpret_cast<const char*>(buf.data()), static_cast<size_t>(flushed));
    if (out.empty()) throw std::runtime_error("MP3 encoder produced empty output");
    return out;
}

static std::string yyjson_to_string(yyjson_mut_doc* doc) {
    size_t len = 0;
    char* raw = yyjson_mut_write(doc, 0, &len);
    if (!raw) return "{}";
    std::string res(raw, len);
    std::free(raw);
    return res;
}

static std::string make_json_error(const std::string& msg) {
    yyjson_mut_doc* doc = yyjson_mut_doc_new(nullptr);
    yyjson_mut_val* root = yyjson_mut_obj(doc);
    yyjson_mut_doc_set_root(doc, root);
    yyjson_mut_obj_add_str(doc, root, "error", msg.c_str());
    std::string s = yyjson_to_string(doc);
    yyjson_mut_doc_free(doc);
    return s;
}

static std::string trim_string(const std::string& s) {
    size_t start = 0;
    while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start]))) ++start;
    size_t end = s.size();
    while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1]))) --end;
    return s.substr(start, end - start);
}

static double clamp_duration(yyjson_val* v) {
    double seconds = 60.0;
    if (v) {
        if (yyjson_is_num(v)) {
            seconds = yyjson_get_num(v);
        } else if (yyjson_is_str(v)) {
            try {
                seconds = std::stod(yyjson_get_str(v));
            } catch (...) {
                seconds = 60.0;
            }
        }
    }
    if (!std::isfinite(seconds) || seconds <= 0.0) seconds = 60.0;
    return std::clamp(seconds, 1.0, static_cast<double>(MAX_SECONDS));
}

static int parse_int_clamped(yyjson_val* v, int fallback, int lo, int hi) {
    if (!v) return fallback;
    double num = fallback;
    if (yyjson_is_int(v)) {
        int64_t iv = yyjson_get_sint(v);
        if (iv < lo) return lo;
        if (iv > hi) return hi;
        return static_cast<int>(iv);
    }
    if (yyjson_is_uint(v)) {
        uint64_t uv = yyjson_get_uint(v);
        if (uv > static_cast<uint64_t>(hi)) return hi;
        int iv = static_cast<int>(uv);
        return std::clamp(iv, lo, hi);
    }
    if (yyjson_is_real(v)) {
        num = yyjson_get_real(v);
    } else if (yyjson_is_str(v)) {
        try {
            num = std::stod(yyjson_get_str(v));
        } catch (...) {
            return fallback;
        }
    } else {
        return fallback;
    }
    if (!std::isfinite(num)) return fallback;
    if (num < static_cast<double>(lo)) return lo;
    if (num > static_cast<double>(hi)) return hi;
    return static_cast<int>(num);
}

static double parse_positive_double(yyjson_val* v, double fallback) {
    if (!v) return fallback;
    double val = fallback;
    if (yyjson_is_num(v)) {
        val = yyjson_get_num(v);
    } else if (yyjson_is_str(v)) {
        try {
            val = std::stod(yyjson_get_str(v));
        } catch (...) {
            return fallback;
        }
    } else {
        return fallback;
    }
    if (!std::isfinite(val) || val <= 0.0) return fallback;
    return val;
}

static int64_t parse_seed(yyjson_val* v) {
    if (!v) return -1;
    if (yyjson_is_int(v)) {
        int64_t iv = yyjson_get_sint(v);
        return iv >= 0 ? (iv % (1LL << 31)) : -1;
    }
    if (yyjson_is_uint(v)) {
        uint64_t uv = yyjson_get_uint(v);
        return static_cast<int64_t>(uv % (1ULL << 31));
    }
    if (yyjson_is_str(v)) {
        try {
            int64_t iv = std::stoll(yyjson_get_str(v));
            return iv >= 0 ? (iv % (1LL << 31)) : -1;
        } catch (...) {
            return -1;
        }
    }
    return -1;
}

struct SynthParams {
    std::string prompt;
    double      duration    = 60.0;
    int         steps       = DEFAULT_STEPS;
    float       cfg         = 1.0f;
    int64_t     seed        = -1;
    int         mp3_bitrate = DEFAULT_BITRATE;
};

struct Job {
    std::string id;
    std::string kind; // "lm" or "synth"
    SynthParams params;
    std::string status = "queued";
    std::string error;
    std::string result;
    std::string content_type = "application/json";
    std::chrono::steady_clock::time_point created_at  = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point finished_at{};
    bool has_finished = false;
    std::atomic<bool> cancel_requested{false};
};

struct Config {
    std::string host        = "127.0.0.1";
    int         port        = 8085;
    std::string models_dir;
    std::string api_key_file;
    std::string variant     = "medium";
    std::string encoding    = "f16";
    std::string t5_encoding = "f16";
    std::string ae_encoding = "f16";
};

class Engine {
public:
    Engine(const Config& cfg, const sa3_api_v1* api)
        : cfg_(cfg), api_(api), models_(discover_models(cfg.models_dir)) {
        worker_ = std::thread([this] { run_worker(); });
    }

    ~Engine() {
        shutdown();
        if (worker_.joinable()) worker_.join();
    }

    const DiscoveredModels& models() const { return models_; }

    std::string props_json() const {
        yyjson_mut_doc* doc = yyjson_mut_doc_new(nullptr);
        yyjson_mut_val* root = yyjson_mut_obj(doc);
        yyjson_mut_doc_set_root(doc, root);

        yyjson_mut_val* m = yyjson_mut_obj(doc);
        auto add_vec = [&](const char* key, const std::vector<std::string>& vals) {
            yyjson_mut_val* arr = yyjson_mut_arr(doc);
            for (const auto& s : vals) yyjson_mut_arr_add_str(doc, arr, s.c_str());
            yyjson_mut_obj_add_val(doc, m, key, arr);
        };
        add_vec("dit", models_.dit);
        add_vec("decoder", models_.decoder);
        add_vec("text_encoder", models_.text_encoder);
        add_vec("tokenizer", models_.tokenizer);
        add_vec("conditioner", models_.conditioner);
        yyjson_mut_obj_add_val(doc, root, "models", m);
        yyjson_mut_obj_add_bool(doc, root, "vocals", false);
        yyjson_mut_obj_add_int(doc, root, "max_seconds", MAX_SECONDS);
        yyjson_mut_obj_add_str(doc, root, "backend", "cuda");
        yyjson_mut_obj_add_str(doc, root, "version", api_->runtime_version());

        std::string s = yyjson_to_string(doc);
        yyjson_mut_doc_free(doc);
        return s;
    }

    std::string submit_lm(yyjson_val* root) {
        std::string caption;
        if (yyjson_val* v = yyjson_obj_get(root, "caption"); v && yyjson_is_str(v)) {
            caption = yyjson_get_str(v);
        } else if (yyjson_val* v2 = yyjson_obj_get(root, "prompt"); v2 && yyjson_is_str(v2)) {
            caption = yyjson_get_str(v2);
        }
        caption = trim_string(caption);
        if (caption.size() > MAX_PROMPT_CHARS) caption.resize(MAX_PROMPT_CHARS);
        double duration = clamp_duration(yyjson_obj_get(root, "duration"));

        yyjson_mut_doc* doc = yyjson_mut_doc_new(nullptr);
        yyjson_mut_val* arr = yyjson_mut_arr(doc);
        yyjson_mut_doc_set_root(doc, arr);
        yyjson_mut_val* item = yyjson_mut_obj(doc);
        yyjson_mut_obj_add_str(doc, item, "caption", caption.c_str());
        yyjson_mut_obj_add_str(doc, item, "lyrics", "[Instrumental]");
        yyjson_mut_obj_add_real(doc, item, "duration", duration);
        yyjson_mut_obj_add_int(doc, item, "bpm", 0);
        yyjson_mut_obj_add_str(doc, item, "keyscale", "");
        yyjson_mut_arr_append(arr, item);
        std::string payload = yyjson_to_string(doc);
        yyjson_mut_doc_free(doc);

        auto job = std::make_shared<Job>();
        job->id = random_hex_id();
        job->kind = "lm";
        job->result = std::move(payload);
        job->content_type = "application/json";
        job->status = "done";
        job->finished_at = std::chrono::steady_clock::now();
        job->has_finished = true;

        {
            std::lock_guard<std::mutex> lk(mu_);
            jobs_[job->id] = job;
            prune_locked();
            cv_.notify_one();
        }
        return job->id;
    }

    bool submit_synth(yyjson_val* root, std::string& out_id, std::string& out_err) {
        SynthParams p;
        if (yyjson_val* v = yyjson_obj_get(root, "caption"); v && yyjson_is_str(v)) {
            p.prompt = yyjson_get_str(v);
        } else if (yyjson_val* v2 = yyjson_obj_get(root, "prompt"); v2 && yyjson_is_str(v2)) {
            p.prompt = yyjson_get_str(v2);
        }
        p.prompt = trim_string(p.prompt);
        if (p.prompt.size() > MAX_PROMPT_CHARS) p.prompt.resize(MAX_PROMPT_CHARS);
        p.duration    = clamp_duration(yyjson_obj_get(root, "duration"));
        p.steps       = parse_int_clamped(yyjson_obj_get(root, "steps"), DEFAULT_STEPS, 1, 32);
        p.cfg         = static_cast<float>(parse_positive_double(yyjson_obj_get(root, "cfg"), 1.0));
        p.seed        = parse_seed(yyjson_obj_get(root, "seed"));
        p.mp3_bitrate = parse_int_clamped(yyjson_obj_get(root, "mp3_bitrate"), DEFAULT_BITRATE, 96, 320);

        auto job = std::make_shared<Job>();
        job->id = random_hex_id();
        job->kind = "synth";
        job->params = std::move(p);

        {
            std::lock_guard<std::mutex> lk(mu_);
            prune_locked();
            size_t pending = 0;
            for (const auto& [_, j] : jobs_) {
                if (j->status == "queued" || j->status == "running") ++pending;
            }
            if (pending >= MAX_PENDING_JOBS) {
                out_err = "at most " + std::to_string(MAX_PENDING_JOBS) + " synthesis jobs may be active or queued";
                return false;
            }
            jobs_[job->id] = job;
            pending_queue_.push_back(job->id);
            cv_.notify_one();
        }
        out_id = job->id;
        return true;
    }

    bool get_job_status(const std::string& id, std::string& status, std::string& error) {
        std::lock_guard<std::mutex> lk(mu_);
        prune_locked();
        auto it = jobs_.find(id);
        if (it == jobs_.end()) return false;
        status = it->second->status;
        error  = it->second->error;
        return true;
    }

    int get_job_result(const std::string& id, std::string& body, std::string& content_type, std::string& status) {
        std::lock_guard<std::mutex> lk(mu_);
        prune_locked();
        auto it = jobs_.find(id);
        if (it == jobs_.end()) return 404;
        status = it->second->status;
        if (status != "done" || it->second->result.empty()) return 409;
        body         = it->second->result;
        content_type = it->second->content_type;
        return 200;
    }

    bool cancel_job(const std::string& id) {
        std::lock_guard<std::mutex> lk(mu_);
        prune_locked();
        auto it = jobs_.find(id);
        if (it == jobs_.end()) return false;
        auto& job = it->second;
        job->cancel_requested.store(true, std::memory_order_release);
        if (job->status == "queued") {
            auto qit = std::find(pending_queue_.begin(), pending_queue_.end(), id);
            if (qit != pending_queue_.end()) pending_queue_.erase(qit);
            job->params = SynthParams{};
            job->status = "cancelled";
            job->finished_at = std::chrono::steady_clock::now();
            job->has_finished = true;
            prune_locked();
            cv_.notify_one();
        }
        return true;
    }

    void shutdown() {
        std::lock_guard<std::mutex> lk(mu_);
        stopping_ = true;
        for (auto& [_, job] : jobs_) {
            job->cancel_requested.store(true, std::memory_order_release);
        }
        pending_queue_.clear();
        cv_.notify_all();
    }

private:
    bool prune_locked() {
        const auto now = std::chrono::steady_clock::now();
        std::vector<std::shared_ptr<Job>> finished;
        for (const auto& [_, job] : jobs_) {
            if (job->has_finished) finished.push_back(job);
        }
        std::sort(finished.begin(), finished.end(), [](const auto& a, const auto& b) {
            return a->finished_at < b->finished_at;
        });
        bool pruned = false;
        size_t remaining = finished.size();
        for (const auto& job : finished) {
            if ((now - job->finished_at) >= JOB_TTL || remaining > MAX_FINISHED_JOBS) {
                jobs_.erase(job->id);
                --remaining;
                pruned = true;
            }
        }
        return pruned;
    }

    bool next_expiry_locked(std::chrono::steady_clock::time_point& out) const {
        bool found = false;
        for (const auto& [_, job] : jobs_) {
            if (!job->has_finished) continue;
            auto exp = job->finished_at + JOB_TTL;
            if (!found || exp < out) {
                out = exp;
                found = true;
            }
        }
        return found;
    }

    static int32_t SA3_CALL cancel_callback(void* user) {
        auto* flag = static_cast<std::atomic<bool>*>(user);
        return (flag && flag->load(std::memory_order_acquire)) ? 1 : 0;
    }

    std::string synthesize(const SynthParams& params, std::atomic<bool>& cancel_flag) {
        if (models_.dit.empty() || models_.decoder.empty() ||
            models_.text_encoder.empty() || models_.tokenizer.empty()) {
            throw std::runtime_error("missing required Stable Audio 3 model files under " + cfg_.models_dir);
        }
        if (cancel_flag.load(std::memory_order_acquire)) {
            throw CancelledError{};
        }

        const auto t0 = std::chrono::steady_clock::now();

        sa3_error_v1 err{};
        err.size = sizeof(err);
        api_->error_init(&err);

        sa3_context_config_v1 ccfg{};
        ccfg.size = sizeof(ccfg);
        api_->context_config_init(&ccfg);
        ccfg.models_dir            = cfg_.models_dir.c_str();
        ccfg.adapters_dir          = cfg_.models_dir.c_str();
        ccfg.variant               = cfg_.variant.c_str();
        ccfg.dit_encoding          = cfg_.encoding.c_str();
        ccfg.text_encoder_encoding = cfg_.t5_encoding.c_str();
        ccfg.autoencoder_encoding  = cfg_.ae_encoding.c_str();
        ccfg.device                = "";

        sa3_context* ctx = nullptr;
        if (api_->context_create(&ccfg, &ctx, &err) != SA3_STATUS_OK_V1) {
            throw std::runtime_error(std::string("context_create failed: ") + err.message);
        }
        struct ContextGuard {
            const sa3_api_v1* api;
            sa3_context* ctx;
            ~ContextGuard() { if (ctx) api->context_destroy(ctx); }
        } ctx_guard{api_, ctx};

        if (cancel_flag.load(std::memory_order_acquire)) {
            throw CancelledError{};
        }

        sa3_request_v1 req{};
        req.size = sizeof(req);
        api_->request_init(&req);
        req.operation        = SA3_OPERATION_GENERATE_V1;
        req.prompt           = params.prompt.c_str();
        req.duration_seconds = params.duration;
        req.steps            = params.steps;
        req.seed             = params.seed >= 0 ? params.seed : random_seed();
        req.cfg_scale        = params.cfg;
        req.residency        = SA3_RESIDENCY_FRUGAL_V1;
        req.should_cancel    = &Engine::cancel_callback;
        req.callback_user    = &cancel_flag;

        sa3_result_v1 result{};
        result.size = sizeof(result);
        api_->result_init(&result);
        struct ResultGuard {
            const sa3_api_v1* api;
            sa3_result_v1* res;
            ~ResultGuard() { api->result_free(res); }
        } res_guard{api_, &result};

        const sa3_status_v1 st = api_->generate(ctx, &req, &result, &err);
        // Release GPU memory immediately once generation completes or aborts.
        api_->context_unload(ctx);

        if (st == SA3_STATUS_CANCELLED_V1 || cancel_flag.load(std::memory_order_acquire)) {
            throw CancelledError{};
        }
        if (st != SA3_STATUS_OK_V1) {
            throw std::runtime_error(std::string("generate failed: ") + err.message);
        }

        std::string mp3 = encode_mp3(result.samples, result.n_samples,
                                     result.n_channels, result.sample_rate,
                                     result.layout, params.mp3_bitrate);
        const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::fprintf(stdout, "synth: %.0fs audio in %.2fs (%.2fx realtime, seed=%llu)\n",
                     params.duration, elapsed, params.duration / std::max(elapsed, 0.01),
                     static_cast<unsigned long long>(result.seed));
        std::fflush(stdout);
        return mp3;
    }

    struct CancelledError {};

    void run_worker() {
        while (true) {
            std::shared_ptr<Job> job;
            SynthParams params;
            {
                std::unique_lock<std::mutex> lk(mu_);
                prune_locked();
                while (!stopping_ && pending_queue_.empty()) {
                    std::chrono::steady_clock::time_point exp;
                    if (next_expiry_locked(exp)) {
                        if (exp <= std::chrono::steady_clock::now()) {
                            prune_locked();
                            continue;
                        }
                        cv_.wait_until(lk, exp);
                    } else {
                        cv_.wait(lk);
                    }
                    prune_locked();
                }
                if (stopping_) return;
                if (pending_queue_.empty()) continue;

                std::string job_id = std::move(pending_queue_.front());
                pending_queue_.pop_front();
                auto it = jobs_.find(job_id);
                if (it == jobs_.end()) continue;
                job = it->second;
                if (job->cancel_requested.load(std::memory_order_acquire) || job->status == "cancelled") {
                    job->params = SynthParams{};
                    job->status = "cancelled";
                    job->finished_at = std::chrono::steady_clock::now();
                    job->has_finished = true;
                    prune_locked();
                    continue;
                }
                job->status = "running";
                params = std::move(job->params);
                job->params = SynthParams{};
            }

            try {
                std::string mp3 = synthesize(params, job->cancel_requested);
                std::string body;
                body.reserve(mp3.size() + 128);
                body += "--";
                body += BOUNDARY;
                body += "\r\nContent-Type: audio/mpeg\r\n\r\n";
                body += mp3;
                body += "\r\n--";
                body += BOUNDARY;
                body += "--\r\n";

                std::lock_guard<std::mutex> lk(mu_);
                job->result = std::move(body);
                job->content_type = std::string("multipart/mixed; boundary=") + BOUNDARY;
                job->status = "done";
                job->finished_at = std::chrono::steady_clock::now();
                job->has_finished = true;
                prune_locked();
            } catch (const CancelledError&) {
                std::lock_guard<std::mutex> lk(mu_);
                job->status = "cancelled";
                job->finished_at = std::chrono::steady_clock::now();
                job->has_finished = true;
                prune_locked();
            } catch (const std::exception& exc) {
                std::fprintf(stderr, "error: synth job %s failed: %s\n", job->id.c_str(), exc.what());
                std::fflush(stderr);
                std::lock_guard<std::mutex> lk(mu_);
                job->error = exc.what();
                job->status = "failed";
                job->finished_at = std::chrono::steady_clock::now();
                job->has_finished = true;
                prune_locked();
            }
        }
    }

    Config cfg_;
    const sa3_api_v1* api_;
    DiscoveredModels models_;
    std::mutex mu_;
    std::condition_variable cv_;
    std::unordered_map<std::string, std::shared_ptr<Job>> jobs_;
    std::deque<std::string> pending_queue_;
    bool stopping_ = false;
    std::thread worker_;
};

static std::string read_api_key(const Config& cfg) {
    if (!cfg.api_key_file.empty()) {
        std::ifstream f(cfg.api_key_file);
        if (!f) return "";
        std::ostringstream oss;
        oss << f.rdbuf();
        return trim_string(oss.str());
    }
    const char* env = std::getenv("SA3_API_KEY");
    return env ? trim_string(env) : "";
}

static void print_usage() {
    std::printf(
        "Usage: sa3-server --models DIR [options]\n"
        "  --host HOST            Bind address (default 127.0.0.1)\n"
        "  --port PORT            Listen port (default 8085)\n"
        "  --models DIR           Directory containing Stable Audio 3 GGUF models\n"
        "  --api-key-file FILE    Read Bearer API key from file (else SA3_API_KEY env)\n"
        "  --variant NAME         Model variant: medium, small-music, small-sfx (default medium)\n"
        "  --encoding ENC         DiT encoding: f16, f32, q8_0, q5_k_m, q4_k_m (default f16)\n"
        "  --t5-encoding ENC      Text encoder encoding: f16, f32, q8_0 (default f16)\n"
        "  --ae-encoding ENC      Autoencoder encoding: f16, f32, q8_0, q5_k_m, q4_k_m (default f16)\n"
        "  --help                 Show this help message\n");
}

} // namespace

int main(int argc, char** argv) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&](const char* flag) -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "error: %s requires a value\n", flag);
                std::exit(2);
            }
            return argv[++i];
        };
        if (arg == "--help" || arg == "-h") {
            print_usage();
            return 0;
        } else if (arg == "--host") {
            cfg.host = next("--host");
        } else if (arg == "--port") {
            cfg.port = std::atoi(next("--port").c_str());
        } else if (arg == "--models") {
            cfg.models_dir = next("--models");
        } else if (arg == "--api-key-file") {
            cfg.api_key_file = next("--api-key-file");
        } else if (arg == "--variant") {
            cfg.variant = next("--variant");
        } else if (arg == "--encoding") {
            cfg.encoding = next("--encoding");
        } else if (arg == "--t5-encoding") {
            cfg.t5_encoding = next("--t5-encoding");
        } else if (arg == "--ae-encoding") {
            cfg.ae_encoding = next("--ae-encoding");
        } else {
            std::fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            return 2;
        }
    }

    if (cfg.models_dir.empty()) {
        std::fprintf(stderr, "error: --models is required\n");
        return 2;
    }
    if (cfg.port <= 0 || cfg.port > 65535) {
        std::fprintf(stderr, "error: invalid --port\n");
        return 2;
    }

    const std::string api_key = read_api_key(cfg);
    if (api_key.empty()) {
        std::fprintf(stderr, "error: SA3_API_KEY or --api-key-file is required\n");
        return 2;
    }
    const std::string expected_auth = "Bearer " + api_key;

    const sa3_api_v1* api = sa3_get_api(SA3_ABI_VERSION_1);
    if (!api) {
        std::fprintf(stderr, "error: libsa3 ABI v1 unavailable\n");
        return 1;
    }

    // Block SIGTERM/SIGINT on all threads (including worker & HTTP pool) and handle
    // them cleanly on a dedicated thread via sigwait.
    sigset_t sigset;
    sigemptyset(&sigset);
    sigaddset(&sigset, SIGTERM);
    sigaddset(&sigset, SIGINT);
    pthread_sigmask(SIG_BLOCK, &sigset, nullptr);

    Engine engine(cfg, api);
    httplib::Server svr;
    svr.set_payload_max_length(MAX_BODY_BYTES);

    auto authorized = [&](const httplib::Request& req, httplib::Response& res) -> bool {
        const std::string hdr = req.get_header_value("Authorization");
        if (!constant_time_equal(hdr, expected_auth)) {
            res.status = 401;
            res.set_content(make_json_error("unauthorized"), "application/json");
            return false;
        }
        return true;
    };

    svr.Get("/health", [&](const httplib::Request& req, httplib::Response& res) {
        if (!authorized(req, res)) return;
        res.status = 200;
        res.set_content("{\"status\":\"ok\"}", "application/json");
    });

    svr.Get("/props", [&](const httplib::Request& req, httplib::Response& res) {
        if (!authorized(req, res)) return;
        res.status = 200;
        res.set_content(engine.props_json(), "application/json");
    });

    svr.Get("/job", [&](const httplib::Request& req, httplib::Response& res) {
        if (!authorized(req, res)) return;
        const std::string id = req.get_param_value("id");
        if (req.has_param("result") && !req.get_param_value("result").empty()) {
            std::string body, ctype, status;
            int code = engine.get_job_result(id, body, ctype, status);
            if (code == 404) {
                res.status = 404;
                res.set_content(make_json_error("unknown job"), "application/json");
                return;
            }
            if (code == 409) {
                res.status = 409;
                res.set_content(make_json_error("job is " + status), "application/json");
                return;
            }
            res.status = 200;
            res.set_content(body, ctype);
            return;
        }
        std::string status, err;
        if (!engine.get_job_status(id, status, err)) {
            res.status = 404;
            res.set_content(make_json_error("unknown job"), "application/json");
            return;
        }
        yyjson_mut_doc* doc = yyjson_mut_doc_new(nullptr);
        yyjson_mut_val* root = yyjson_mut_obj(doc);
        yyjson_mut_doc_set_root(doc, root);
        yyjson_mut_obj_add_str(doc, root, "id", id.c_str());
        yyjson_mut_obj_add_str(doc, root, "status", status.c_str());
        if (!err.empty()) yyjson_mut_obj_add_str(doc, root, "error", err.c_str());
        res.status = 200;
        res.set_content(yyjson_to_string(doc), "application/json");
        yyjson_mut_doc_free(doc);
    });

    auto parse_json_obj = [&](const httplib::Request& req, httplib::Response& res) -> yyjson_doc* {
        if (req.body.size() > MAX_BODY_BYTES) {
            res.status = 400;
            res.set_content(make_json_error("body too large"), "application/json");
            return nullptr;
        }
        const std::string body = req.body.empty() ? "{}" : req.body;
        yyjson_doc* doc = yyjson_read(body.data(), body.size(), 0);
        if (!doc || !yyjson_is_obj(yyjson_doc_get_root(doc))) {
            if (doc) yyjson_doc_free(doc);
            res.status = 400;
            res.set_content(make_json_error("invalid json"), "application/json");
            return nullptr;
        }
        return doc;
    };

    svr.Post("/lm", [&](const httplib::Request& req, httplib::Response& res) {
        if (!authorized(req, res)) return;
        yyjson_doc* doc = parse_json_obj(req, res);
        if (!doc) return;
        std::string id = engine.submit_lm(yyjson_doc_get_root(doc));
        yyjson_doc_free(doc);

        yyjson_mut_doc* out = yyjson_mut_doc_new(nullptr);
        yyjson_mut_val* root = yyjson_mut_obj(out);
        yyjson_mut_doc_set_root(out, root);
        yyjson_mut_obj_add_str(out, root, "id", id.c_str());
        res.status = 200;
        res.set_content(yyjson_to_string(out), "application/json");
        yyjson_mut_doc_free(out);
    });

    svr.Post("/synth", [&](const httplib::Request& req, httplib::Response& res) {
        if (!authorized(req, res)) return;
        yyjson_doc* doc = parse_json_obj(req, res);
        if (!doc) return;
        std::string id, err;
        bool ok = engine.submit_synth(yyjson_doc_get_root(doc), id, err);
        yyjson_doc_free(doc);
        if (!ok) {
            res.status = 429;
            res.set_content(make_json_error(err), "application/json");
            return;
        }
        yyjson_mut_doc* out = yyjson_mut_doc_new(nullptr);
        yyjson_mut_val* root = yyjson_mut_obj(out);
        yyjson_mut_doc_set_root(out, root);
        yyjson_mut_obj_add_str(out, root, "id", id.c_str());
        res.status = 200;
        res.set_content(yyjson_to_string(out), "application/json");
        yyjson_mut_doc_free(out);
    });

    svr.Post("/job", [&](const httplib::Request& req, httplib::Response& res) {
        if (!authorized(req, res)) return;
        const std::string id = req.get_param_value("id");
        if (req.has_param("cancel") && !req.get_param_value("cancel").empty()) {
            if (!engine.cancel_job(id)) {
                res.status = 404;
                res.set_content(make_json_error("unknown job"), "application/json");
                return;
            }
            res.status = 200;
            res.set_content("{\"cancelled\":true}", "application/json");
            return;
        }
        res.status = 404;
        res.set_content(make_json_error("not found"), "application/json");
    });

    svr.set_error_handler([&](const httplib::Request& req, httplib::Response& res) {
        if (res.body.empty()) {
            if (!constant_time_equal(req.get_header_value("Authorization"), expected_auth)) {
                res.status = 401;
                res.set_content(make_json_error("unauthorized"), "application/json");
            } else {
                res.set_content(make_json_error("not found"), "application/json");
            }
        }
    });

    std::atomic<bool> sig_stop{false};
    std::thread sig_thread([&] {
        int sig = 0;
        if (sigwait(&sigset, &sig) == 0) {
            sig_stop.store(true, std::memory_order_release);
            engine.shutdown();
            svr.stop();
        }
    });

    std::fprintf(stdout, "sa3-server: listening on %s:%d (variant=%s, encoding=%s, backend=cuda, models=%zu/%zu/%zu/%zu)\n",
                 cfg.host.c_str(), cfg.port, cfg.variant.c_str(), cfg.encoding.c_str(),
                 engine.models().dit.size(), engine.models().decoder.size(),
                 engine.models().text_encoder.size(), engine.models().tokenizer.size());
    std::fflush(stdout);

    const bool ok = svr.listen(cfg.host, cfg.port);
    if (!sig_stop.load(std::memory_order_acquire)) {
        // Trigger sigwait thread to exit cleanly on listen failure.
        pthread_kill(sig_thread.native_handle(), SIGTERM);
    }
    if (sig_thread.joinable()) sig_thread.join();
    engine.shutdown();
    return (ok || sig_stop.load(std::memory_order_acquire)) ? 0 : 1;
}
