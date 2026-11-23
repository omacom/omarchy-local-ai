// sa3-server: HTTP server for Stable Audio 3 GGUF models via sa3.cpp (GGML Vulkan),
// speaking ace-server's async job protocol for omarchy-local-ai.
//
// Every route requires `Authorization: Bearer <key>` matched in constant time against
// SA3_API_KEY or --api-key-file. Weights live read-only under --models; each stage
// (T5Gemma, DiT, SAME-L decoder) loads only while a song needs it and the backend is
// released after every render so the server holds zero GPU memory between songs.

#include "sa3_pipeline.h"

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
#include <limits>
#include <memory>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace {

constexpr int    kSampleRate       = 44100;
constexpr int    kDefaultSteps     = 8;
constexpr int    kDefaultBitrate   = 192;
constexpr int    kDefaultMaxSec    = 180;
constexpr int    kDefaultChunkSize = 680;
constexpr int    kDecodeOverlap    = 32;
constexpr size_t kMaxBodyBytes     = 1u << 20;
constexpr size_t kMaxPromptChars   = 4096;
constexpr size_t kMaxPendingJobs   = 8;
constexpr size_t kMaxFinishedJobs  = 8;
constexpr double kJobTtlSeconds    = 600.0;
constexpr char   kBoundary[]       = "sa3-batch-boundary";

struct DiscoveredModels {
    std::vector<std::string> dit;
    std::vector<std::string> decoder;
    std::vector<std::string> text_encoder;
    std::vector<std::string> tokenizer;
    std::vector<std::string> conditioner;
};

static std::string to_lower_ascii(std::string s) {
    for (char& c : s) {
        if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
    }
    return s;
}

static std::string trim_whitespace(std::string s) {
    auto is_ws = [](unsigned char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\n'; };
    while (!s.empty() && is_ws(static_cast<unsigned char>(s.front()))) s.erase(s.begin());
    while (!s.empty() && is_ws(static_cast<unsigned char>(s.back()))) s.pop_back();
    return s;
}

static bool ends_with(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

static DiscoveredModels discover_models(const std::filesystem::path& models_dir) {
    namespace fs = std::filesystem;
    DiscoveredModels found;
    std::error_code ec;
    if (!fs::is_directory(models_dir, ec)) return found;

    std::vector<fs::path> files;
    for (const auto& entry : fs::recursive_directory_iterator(models_dir, ec)) {
        if (entry.is_regular_file(ec)) files.push_back(entry.path());
    }
    std::sort(files.begin(), files.end());

    for (const auto& path : files) {
        const std::string rel  = fs::relative(path, models_dir, ec).generic_string();
        const std::string name = to_lower_ascii(path.filename().string());
        if (!ends_with(name, ".gguf")) continue;
        if (name.find("-dit-") != std::string::npos || name.rfind("dit", 0) == 0) {
            found.dit.push_back(rel);
        } else if (name.find("-same-") != std::string::npos || name.rfind("dec", 0) == 0) {
            found.decoder.push_back(rel);
        } else if (name.find("vocab") != std::string::npos || name.find("tokenizer") != std::string::npos) {
            found.tokenizer.push_back(rel);
        } else if (name.find("conditioner") != std::string::npos) {
            found.conditioner.push_back(rel);
        } else if (name.find("t5gemma") != std::string::npos || name.find("encoder") != std::string::npos) {
            found.text_encoder.push_back(rel);
        }
    }
    return found;
}

// Constant-time comparison of `Authorization: Bearer <key>`.
static bool authorized(const httplib::Request& req, const std::string& key) {
    const std::string header = req.get_header_value("Authorization");
    const std::string prefix = "Bearer ";
    if (header.size() != prefix.size() + key.size() || header.compare(0, prefix.size(), prefix) != 0) {
        return false;
    }
    unsigned char diff = 0;
    for (size_t i = 0; i < key.size(); ++i) {
        diff |= static_cast<unsigned char>(header[prefix.size() + i] ^ key[i]);
    }
    return diff == 0;
}

static std::string json_string(yyjson_mut_doc* doc) {
    size_t len = 0;
    char* raw = yyjson_mut_write(doc, 0, &len);
    std::string out(raw ? raw : "{}", raw ? len : 2);
    free(raw);
    return out;
}

static void send_json_error(httplib::Response& res, int status, const std::string& msg) {
    yyjson_mut_doc* doc = yyjson_mut_doc_new(nullptr);
    yyjson_mut_val* root = yyjson_mut_obj(doc);
    yyjson_mut_doc_set_root(doc, root);
    yyjson_mut_obj_add_strcpy(doc, root, "error", msg.c_str());
    res.status = status;
    res.set_content(json_string(doc), "application/json");
    yyjson_mut_doc_free(doc);
}

static std::string random_hex_id() {
    static thread_local std::mt19937_64 rng([] {
        std::random_device rd;
        return (static_cast<uint64_t>(rd()) << 32) ^ static_cast<uint64_t>(rd()) ^
               static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());
    }());
    const uint64_t v = rng();
    char buf[17];
    std::snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(v));
    return std::string(buf, 16);
}

static std::string encode_mp3_planar(const float* samples, int n_samp, int n_ch, int sample_rate, int bitrate_kbps) {
    if (!samples || n_samp <= 0 || n_ch < 1) {
        throw std::runtime_error("no audio samples produced");
    }
    lame_t gfp = lame_init();
    if (!gfp) throw std::runtime_error("lame_init failed");
    struct LameGuard {
        lame_t g;
        ~LameGuard() { if (g) lame_close(g); }
    } guard{gfp};

    lame_set_num_channels(gfp, 2);
    lame_set_in_samplerate(gfp, sample_rate);
    lame_set_brate(gfp, bitrate_kbps);
    lame_set_mode(gfp, JOINT_STEREO);
    lame_set_quality(gfp, 2);
    lame_set_bWriteVbrTag(gfp, 0);
    if (lame_init_params(gfp) < 0) {
        throw std::runtime_error("lame_init_params failed");
    }

    const float* left  = samples;
    const float* right = (n_ch >= 2) ? (samples + n_samp) : samples;

    // Encode in bounded chunks so temporary buffer stays small.
    constexpr int kChunk = 16384;
    std::vector<float> clip_l(kChunk), clip_r(kChunk);
    std::vector<unsigned char> mp3_chunk(static_cast<size_t>(kChunk) * 5 / 4 + 7200);
    std::string out;
    out.reserve(static_cast<size_t>(n_samp) * static_cast<size_t>(bitrate_kbps) / (8 * sample_rate) + 16384);

    for (int pos = 0; pos < n_samp; pos += kChunk) {
        const int take = std::min(kChunk, n_samp - pos);
        for (int i = 0; i < take; ++i) {
            float l = left[pos + i];
            float r = right[pos + i];
            clip_l[i] = std::isfinite(l) ? std::clamp(l, -1.0f, 1.0f) : 0.0f;
            clip_r[i] = std::isfinite(r) ? std::clamp(r, -1.0f, 1.0f) : 0.0f;
        }
        const int written = lame_encode_buffer_ieee_float(
            gfp, clip_l.data(), clip_r.data(), take, mp3_chunk.data(), static_cast<int>(mp3_chunk.size()));
        if (written < 0) {
            throw std::runtime_error("lame_encode_buffer_ieee_float failed (" + std::to_string(written) + ")");
        }
        if (written > 0) {
            out.append(reinterpret_cast<const char*>(mp3_chunk.data()), static_cast<size_t>(written));
        }
    }
    const int flushed = lame_encode_flush(gfp, mp3_chunk.data(), static_cast<int>(mp3_chunk.size()));
    if (flushed < 0) {
        throw std::runtime_error("lame_encode_flush failed (" + std::to_string(flushed) + ")");
    }
    if (flushed > 0) {
        out.append(reinterpret_cast<const char*>(mp3_chunk.data()), static_cast<size_t>(flushed));
    }
    if (out.empty()) {
        throw std::runtime_error("lame produced empty MP3 output");
    }
    return out;
}

struct SynthParams {
    std::string prompt;
    double      duration    = 60.0;
    int         steps       = kDefaultSteps;
    float       cfg         = 1.0f;
    long long   seed        = -1;
    int         mp3_bitrate = kDefaultBitrate;
};

struct Job {
    std::string id;
    std::string kind;
    SynthParams params;
    std::string status = "queued";
    std::string error;
    std::string result;
    std::string content_type = "application/json";
    std::chrono::steady_clock::time_point created_at  = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point finished_at = {};
    std::atomic<bool> cancelled{false};
};

static bool yyjson_get_double_relaxed(yyjson_val* obj, const char* key, double& out) {
    yyjson_val* v = yyjson_obj_get(obj, key);
    if (!v || yyjson_is_null(v)) return false;
    if (yyjson_is_real(v)) {
        out = yyjson_get_real(v);
        return true;
    }
    if (yyjson_is_sint(v)) {
        out = static_cast<double>(yyjson_get_sint(v));
        return true;
    }
    if (yyjson_is_uint(v)) {
        out = static_cast<double>(yyjson_get_uint(v));
        return true;
    }
    if (yyjson_is_str(v)) {
        const char* s = yyjson_get_str(v);
        if (!s || !*s) return false;
        char* end = nullptr;
        double parsed = std::strtod(s, &end);
        if (end && *end == '\0') {
            out = parsed;
            return true;
        }
    }
    return false;
}

static bool yyjson_get_int64_relaxed(yyjson_val* obj, const char* key, int64_t& out) {
    yyjson_val* v = yyjson_obj_get(obj, key);
    if (!v || yyjson_is_null(v) || yyjson_is_bool(v)) return false;
    if (yyjson_is_sint(v)) {
        out = yyjson_get_sint(v);
        return true;
    }
    if (yyjson_is_uint(v)) {
        const uint64_t u = yyjson_get_uint(v);
        out = (u <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
                  ? static_cast<int64_t>(u)
                  : std::numeric_limits<int64_t>::max();
        return true;
    }
    if (yyjson_is_real(v)) {
        const double d = yyjson_get_real(v);
        if (!std::isfinite(d) || d < -1e18 || d > 1e18) return false;
        out = static_cast<int64_t>(d);
        return true;
    }
    if (yyjson_is_str(v)) {
        const char* s = yyjson_get_str(v);
        if (!s || !*s) return false;
        char* end = nullptr;
        errno = 0;
        long long parsed = std::strtoll(s, &end, 10);
        if (errno == 0 && end && *end == '\0') {
            out = static_cast<int64_t>(parsed);
            return true;
        }
    }
    return false;
}

static std::string extract_prompt(yyjson_val* root) {
    yyjson_val* v = yyjson_obj_get(root, "caption");
    if (!v || !yyjson_is_str(v) || yyjson_get_len(v) == 0) {
        v = yyjson_obj_get(root, "prompt");
    }
    if (!v || !yyjson_is_str(v)) return "";
    std::string s(yyjson_get_str(v), yyjson_get_len(v));
    s = trim_whitespace(std::move(s));
    if (s.size() > kMaxPromptChars) s.resize(kMaxPromptChars);
    return s;
}

class Engine {
public:
    Engine(std::filesystem::path models_dir, int max_seconds, int decode_chunk_size)
        : models_dir_(std::move(models_dir)),
          max_seconds_(std::max(1, max_seconds)),
          decode_chunk_size_(std::max(0, decode_chunk_size)),
          models_(discover_models(models_dir_)) {
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

        yyjson_mut_val* models_obj = yyjson_mut_obj(doc);
        auto add_list = [&](const char* key, const std::vector<std::string>& items) {
            yyjson_mut_val* arr = yyjson_mut_arr(doc);
            for (const auto& item : items) {
                yyjson_mut_arr_add_strcpy(doc, arr, item.c_str());
            }
            yyjson_mut_obj_add_val(doc, models_obj, key, arr);
        };
        add_list("dit", models_.dit);
        add_list("decoder", models_.decoder);
        add_list("text_encoder", models_.text_encoder);
        add_list("tokenizer", models_.tokenizer);

        yyjson_mut_obj_add_val(doc, root, "models", models_obj);
        yyjson_mut_obj_add_bool(doc, root, "vocals", false);
        yyjson_mut_obj_add_int(doc, root, "max_seconds", max_seconds_);
        yyjson_mut_obj_add_str(doc, root, "backend", "vulkan");

        std::string out = json_string(doc);
        yyjson_mut_doc_free(doc);
        return out;
    }

    double clamp_duration(double raw) const {
        if (!std::isfinite(raw) || raw <= 0.0) raw = 60.0;
        return std::clamp(raw, 1.0, static_cast<double>(max_seconds_));
    }

    SynthParams sanitize_synth_params(yyjson_val* root) const {
        SynthParams p;
        p.prompt = extract_prompt(root);

        double dur = 60.0;
        if (yyjson_get_double_relaxed(root, "duration", dur)) {
            p.duration = clamp_duration(dur);
        } else {
            p.duration = clamp_duration(60.0);
        }

        int64_t steps = kDefaultSteps;
        if (yyjson_get_int64_relaxed(root, "steps", steps)) {
            p.steps = static_cast<int>(std::clamp<int64_t>(steps, 1, 32));
        }

        double cfg = 1.0;
        if (yyjson_get_double_relaxed(root, "cfg", cfg) && std::isfinite(cfg) && cfg > 0.0 && cfg <= 30.0) {
            p.cfg = static_cast<float>(cfg);
        }

        int64_t seed = -1;
        if (yyjson_get_int64_relaxed(root, "seed", seed) && seed >= 0) {
            p.seed = static_cast<long long>(seed % (1LL << 31));
        }

        int64_t bitrate = kDefaultBitrate;
        if (yyjson_get_int64_relaxed(root, "mp3_bitrate", bitrate)) {
            p.mp3_bitrate = static_cast<int>(std::clamp<int64_t>(bitrate, 96, 320));
        }
        return p;
    }

    std::shared_ptr<Job> submit_lm(yyjson_val* root) {
        auto job = std::make_shared<Job>();
        job->id = random_hex_id();
        job->kind = "lm";
        const std::string caption = extract_prompt(root);
        double dur = 60.0;
        yyjson_get_double_relaxed(root, "duration", dur);
        dur = clamp_duration(dur);

        yyjson_mut_doc* doc = yyjson_mut_doc_new(nullptr);
        yyjson_mut_val* arr = yyjson_mut_arr(doc);
        yyjson_mut_doc_set_root(doc, arr);
        yyjson_mut_val* rec = yyjson_mut_obj(doc);
        yyjson_mut_obj_add_strcpy(doc, rec, "caption", caption.c_str());
        yyjson_mut_obj_add_str(doc, rec, "lyrics", "[Instrumental]");
        yyjson_mut_obj_add_real(doc, rec, "duration", dur);
        yyjson_mut_obj_add_int(doc, rec, "bpm", 0);
        yyjson_mut_obj_add_str(doc, rec, "keyscale", "");
        yyjson_mut_arr_append(arr, rec);

        job->result = json_string(doc);
        yyjson_mut_doc_free(doc);
        job->content_type = "application/json";
        job->status = "done";
        job->finished_at = std::chrono::steady_clock::now();

        {
            std::lock_guard<std::mutex> lk(mu_);
            jobs_[job->id] = job;
            prune_locked();
        }
        cv_.notify_one();
        return job;
    }

    std::shared_ptr<Job> submit_synth(yyjson_val* root, std::string& queue_err) {
        auto job = std::make_shared<Job>();
        job->id = random_hex_id();
        job->kind = "synth";
        job->params = sanitize_synth_params(root);

        {
            std::lock_guard<std::mutex> lk(mu_);
            prune_locked();
            size_t pending = 0;
            for (const auto& kv : jobs_) {
                if (kv.second->status == "queued" || kv.second->status == "running") ++pending;
            }
            if (pending >= kMaxPendingJobs) {
                queue_err = "at most " + std::to_string(kMaxPendingJobs) +
                            " synthesis jobs may be active or queued";
                return nullptr;
            }
            jobs_[job->id] = job;
            pending_queue_.push_back(job->id);
        }
        cv_.notify_one();
        return job;
    }

    struct JobSnapshot {
        bool found = false;
        std::string id;
        std::string status;
        std::string error;
        std::string result;
        std::string content_type;
    };

    JobSnapshot snapshot_job(const std::string& id, bool include_result) {
        std::lock_guard<std::mutex> lk(mu_);
        prune_locked();
        auto it = jobs_.find(id);
        if (it == jobs_.end()) return {};
        const auto& job = it->second;
        JobSnapshot snap;
        snap.found = true;
        snap.id = job->id;
        snap.status = job->status;
        snap.error = job->error;
        if (include_result && job->status == "done") {
            snap.result = job->result;
            snap.content_type = job->content_type;
        }
        return snap;
    }

    bool cancel_job(const std::string& id) {
        std::lock_guard<std::mutex> lk(mu_);
        prune_locked();
        auto it = jobs_.find(id);
        if (it == jobs_.end()) return false;
        auto& job = it->second;
        job->cancelled.store(true, std::memory_order_relaxed);
        if (job->status == "queued") {
            auto qit = std::find(pending_queue_.begin(), pending_queue_.end(), id);
            if (qit != pending_queue_.end()) pending_queue_.erase(qit);
            job->params = {};
            job->status = "cancelled";
            job->finished_at = std::chrono::steady_clock::now();
            prune_locked();
            cv_.notify_one();
        }
        return true;
    }

    void shutdown() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            stopping_ = true;
            for (auto& kv : jobs_) {
                kv.second->cancelled.store(true, std::memory_order_relaxed);
            }
            pending_queue_.clear();
        }
        cv_.notify_all();
    }

private:
    static bool is_finished(const std::string& status) {
        return status == "done" || status == "failed" || status == "cancelled";
    }

    void prune_locked() {
        const auto now = std::chrono::steady_clock::now();
        std::vector<std::shared_ptr<Job>> finished;
        for (const auto& kv : jobs_) {
            if (is_finished(kv.second->status)) finished.push_back(kv.second);
        }
        std::sort(finished.begin(), finished.end(), [](const auto& a, const auto& b) {
            return a->finished_at < b->finished_at;
        });
        size_t remaining = finished.size();
        for (const auto& job : finished) {
            const double age = std::chrono::duration<double>(now - job->finished_at).count();
            if (age >= kJobTtlSeconds || remaining > kMaxFinishedJobs) {
                jobs_.erase(job->id);
                --remaining;
            }
        }
    }

    bool next_expiry_locked(std::chrono::steady_clock::time_point& out) const {
        bool found = false;
        for (const auto& kv : jobs_) {
            if (!is_finished(kv.second->status)) continue;
            const auto exp = kv.second->finished_at +
                             std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                                 std::chrono::duration<double>(kJobTtlSeconds));
            if (!found || exp < out) {
                out = exp;
                found = true;
            }
        }
        return found;
    }

    void run_worker() {
        while (true) {
            std::shared_ptr<Job> job;
            SynthParams params;
            {
                std::unique_lock<std::mutex> lk(mu_);
                prune_locked();
                while (!stopping_ && pending_queue_.empty()) {
                    std::chrono::steady_clock::time_point expiry;
                    if (next_expiry_locked(expiry)) {
                        if (expiry <= std::chrono::steady_clock::now()) {
                            prune_locked();
                            continue;
                        }
                        cv_.wait_until(lk, expiry);
                    } else {
                        cv_.wait(lk);
                    }
                    prune_locked();
                }
                if (stopping_) return;
                if (pending_queue_.empty()) continue;

                const std::string id = pending_queue_.front();
                pending_queue_.pop_front();
                auto it = jobs_.find(id);
                if (it == jobs_.end()) continue;
                job = it->second;
                if (job->cancelled.load(std::memory_order_relaxed) || job->status == "cancelled") {
                    job->params = {};
                    job->status = "cancelled";
                    job->finished_at = std::chrono::steady_clock::now();
                    prune_locked();
                    continue;
                }
                job->status = "running";
                params = std::move(job->params);
                job->params = {};
            }

            try {
                std::string mp3 = synthesize(params, job->cancelled);
                std::string body;
                body.reserve(mp3.size() + 128);
                body += "--";
                body += kBoundary;
                body += "\r\nContent-Type: audio/mpeg\r\n\r\n";
                body += mp3;
                body += "\r\n--";
                body += kBoundary;
                body += "--\r\n";

                std::lock_guard<std::mutex> lk(mu_);
                job->result = std::move(body);
                job->content_type = std::string("multipart/mixed; boundary=") + kBoundary;
                job->status = "done";
                job->finished_at = std::chrono::steady_clock::now();
                prune_locked();
            } catch (const std::exception& exc) {
                const bool was_cancelled = job->cancelled.load(std::memory_order_relaxed) ||
                                           std::string(exc.what()).find("cancelled") != std::string::npos;
                if (!was_cancelled) {
                    std::fprintf(stderr, "error: synth job %s failed: %s\n", job->id.c_str(), exc.what());
                    std::fflush(stderr);
                }
                std::lock_guard<std::mutex> lk(mu_);
                if (was_cancelled) {
                    job->status = "cancelled";
                } else {
                    job->error = exc.what();
                    job->status = "failed";
                }
                job->finished_at = std::chrono::steady_clock::now();
                prune_locked();
            }
        }
    }

    std::string synthesize(const SynthParams& params, const std::atomic<bool>& cancelled) const {
        if (models_.dit.empty() || models_.decoder.empty() ||
            models_.text_encoder.empty() || models_.tokenizer.empty()) {
            throw std::runtime_error("missing required Stable Audio 3 model files under --models");
        }
        if (cancelled.load(std::memory_order_relaxed)) {
            throw std::runtime_error("generation cancelled");
        }

        const auto t0 = std::chrono::steady_clock::now();

        sa3::ModelPaths paths;
        paths.dit  = (models_dir_ / models_.dit.front()).string();
        paths.same = (models_dir_ / models_.decoder.front()).string();
        paths.t5   = (models_dir_ / models_.text_encoder.front()).string();
        paths.tok  = (models_dir_ / models_.tokenizer.front()).string();
        if (!models_.conditioner.empty()) {
            paths.cond = (models_dir_ / models_.conditioner.front()).string();
        }

        const int target_n_samp = std::max(1, static_cast<int>(std::llround(params.duration * kSampleRate)));
        int frames = std::max(1, (target_n_samp + 4095) / 4096);
        if (frames & 1) ++frames;

        sa3::GenParams gp;
        gp.prompt = params.prompt;
        gp.frames = frames;
        gp.target_n_samp = target_n_samp;
        gp.steps = params.steps;
        gp.seed = sa3::pick_seed(params.seed);
        gp.cfg_scale = params.cfg;
        gp.apg_scale = 1.0f;
        gp.duration_padding_sec = 0.0f;
        gp.keep_models = false;
        gp.decode_chunk_size = (decode_chunk_size_ > 0 && frames > decode_chunk_size_) ? decode_chunk_size_ : 0;
        gp.decode_overlap = kDecodeOverlap;
        gp.should_cancel = [&]() { return cancelled.load(std::memory_order_relaxed); };

        sa3::GenResult res;
        {
            // Scope the Pipeline so its Vulkan backend and any transient device pools
            // are released completely when synthesis ends, leaving 0 GiB GPU memory idle.
            sa3::Pipeline pipe;
            pipe.load(paths);
            res = pipe.generate(gp);
        }

        if (cancelled.load(std::memory_order_relaxed)) {
            throw std::runtime_error("generation cancelled");
        }

        const int n_samp = std::min(res.n_samp, target_n_samp);
        std::string mp3 = encode_mp3_planar(res.samples.data(), n_samp, res.n_ch, res.sample_rate, params.mp3_bitrate);
        const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::fprintf(stderr, "synth: %.0fs audio in %.2fs (%.2fx realtime, seed=%llu, chunk=%d)\n",
                     params.duration, elapsed, params.duration / std::max(elapsed, 0.01),
                     static_cast<unsigned long long>(gp.seed), gp.decode_chunk_size);
        std::fflush(stderr);
        return mp3;
    }

    std::filesystem::path models_dir_;
    int max_seconds_;
    int decode_chunk_size_;
    DiscoveredModels models_;

    std::mutex mu_;
    std::condition_variable cv_;
    std::unordered_map<std::string, std::shared_ptr<Job>> jobs_;
    std::deque<std::string> pending_queue_;
    bool stopping_ = false;
    std::thread worker_;
};

static std::atomic<bool> g_stop_requested{false};

static void handle_signal(int) {
    g_stop_requested.store(true, std::memory_order_relaxed);
}

static void print_usage(const char* prog) {
    std::fprintf(stderr,
                 "usage: %s --models <dir> [--host <addr>] [--port <N>]\n"
                 "          [--api-key <key> | --api-key-file <path>]\n"
                 "          [--decode-chunk-size <frames>] [--max-seconds <N>]\n"
                 "\n"
                 "API key is required via SA3_API_KEY, --api-key-file, or --api-key.\n",
                 prog);
}

static std::string read_key(const std::string& api_key_arg, const std::string& api_key_file) {
    if (!api_key_arg.empty() && !api_key_file.empty()) {
        std::fprintf(stderr, "error: use --api-key or --api-key-file, not both\n");
        std::exit(2);
    }
    if (!api_key_file.empty()) {
        std::ifstream f(api_key_file);
        std::string line;
        std::getline(f, line);
        return trim_whitespace(std::move(line));
    }
    if (!api_key_arg.empty()) return trim_whitespace(api_key_arg);
    if (const char* env = std::getenv("SA3_API_KEY")) return trim_whitespace(env);
    return "";
}

} // namespace

int main(int argc, char** argv) {
    std::string host = "127.0.0.1";
    int port = 8085;
    std::string models_dir;
    std::string api_key_arg;
    std::string api_key_file;
    int decode_chunk_size = kDefaultChunkSize;
    int max_seconds = kDefaultMaxSec;

    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--help") || !std::strcmp(argv[i], "-h")) {
            print_usage(argv[0]);
            return 0;
        } else if (!std::strcmp(argv[i], "--host") && i + 1 < argc) {
            host = argv[++i];
        } else if (!std::strcmp(argv[i], "--port") && i + 1 < argc) {
            port = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--models") && i + 1 < argc) {
            models_dir = argv[++i];
        } else if (!std::strcmp(argv[i], "--api-key") && i + 1 < argc) {
            api_key_arg = argv[++i];
        } else if (!std::strcmp(argv[i], "--api-key-file") && i + 1 < argc) {
            api_key_file = argv[++i];
        } else if (!std::strcmp(argv[i], "--decode-chunk-size") && i + 1 < argc) {
            decode_chunk_size = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--max-seconds") && i + 1 < argc) {
            max_seconds = std::atoi(argv[++i]);
        } else {
            std::fprintf(stderr, "error: unknown or incomplete option '%s'\n", argv[i]);
            print_usage(argv[0]);
            return 2;
        }
    }

    if (models_dir.empty()) {
        std::fprintf(stderr, "error: --models is required\n");
        print_usage(argv[0]);
        return 2;
    }

    const std::string api_key = read_key(api_key_arg, api_key_file);
    if (api_key.empty()) {
        std::fprintf(stderr, "error: SA3_API_KEY or --api-key-file is required\n");
        return 2;
    }

    // Optimal AMD Vulkan defaults unless explicitly overridden by the environment:
    // DiT + T5Gemma use fused flash attention; SAME-L uses its linear sliding-window path
    // without allocating a quadratic full band mask.
    setenv("SA3_FLASH_ATTN", "1", 0);
    setenv("SA3_SAME_FLASH_ATTN", "0", 0);

    Engine engine(models_dir, max_seconds, decode_chunk_size);
    std::fprintf(stderr, "sa3-server: listening on %s:%d (dit=%zu, decoder=%zu, text_encoder=%zu, tokenizer=%zu)\n",
                 host.c_str(), port, engine.models().dit.size(), engine.models().decoder.size(),
                 engine.models().text_encoder.size(), engine.models().tokenizer.size());
    std::fflush(stderr);

    httplib::Server svr;
    svr.set_payload_max_length(kMaxBodyBytes);
    svr.set_keep_alive_max_count(16);
    svr.set_keep_alive_timeout(15);
    svr.set_read_timeout(60);
    svr.set_write_timeout(600);

    svr.set_pre_routing_handler([&api_key](const httplib::Request& req, httplib::Response& res) {
        if (authorized(req, api_key)) {
            return httplib::Server::HandlerResponse::Unhandled;
        }
        send_json_error(res, 401, "unauthorized");
        return httplib::Server::HandlerResponse::Handled;
    });

    svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_content("{\"status\":\"ok\"}", "application/json");
    });

    svr.Get("/props", [&engine](const httplib::Request&, httplib::Response& res) {
        res.set_content(engine.props_json(), "application/json");
    });

    svr.Get("/job", [&engine](const httplib::Request& req, httplib::Response& res) {
        const std::string id = req.get_param_value("id");
        const bool want_result = req.has_param("result") && !req.get_param_value("result").empty();
        auto snap = engine.snapshot_job(id, want_result);
        if (!snap.found) {
            send_json_error(res, 404, "unknown job");
            return;
        }
        if (want_result) {
            if (snap.status != "done" || snap.result.empty()) {
                send_json_error(res, 409, "job is " + snap.status);
                return;
            }
            res.set_content(snap.result, snap.content_type);
            return;
        }
        yyjson_mut_doc* doc = yyjson_mut_doc_new(nullptr);
        yyjson_mut_val* root = yyjson_mut_obj(doc);
        yyjson_mut_doc_set_root(doc, root);
        yyjson_mut_obj_add_strcpy(doc, root, "id", snap.id.c_str());
        yyjson_mut_obj_add_strcpy(doc, root, "status", snap.status.c_str());
        if (!snap.error.empty()) {
            yyjson_mut_obj_add_strcpy(doc, root, "error", snap.error.c_str());
        }
        res.set_content(json_string(doc), "application/json");
        yyjson_mut_doc_free(doc);
    });

    auto parse_json_object = [](const httplib::Request& req, yyjson_doc*& doc_out) -> yyjson_val* {
        if (req.body.empty()) {
            doc_out = yyjson_read("{}", 2, 0);
        } else {
            doc_out = yyjson_read(req.body.data(), req.body.size(), 0);
        }
        if (!doc_out) return nullptr;
        yyjson_val* root = yyjson_doc_get_root(doc_out);
        if (!root || !yyjson_is_obj(root)) {
            yyjson_doc_free(doc_out);
            doc_out = nullptr;
            return nullptr;
        }
        return root;
    };

    svr.Post("/lm", [&engine, &parse_json_object](const httplib::Request& req, httplib::Response& res) {
        yyjson_doc* doc = nullptr;
        yyjson_val* root = parse_json_object(req, doc);
        if (!root) {
            send_json_error(res, 400, "invalid json");
            return;
        }
        auto job = engine.submit_lm(root);
        yyjson_doc_free(doc);

        yyjson_mut_doc* out = yyjson_mut_doc_new(nullptr);
        yyjson_mut_val* obj = yyjson_mut_obj(out);
        yyjson_mut_doc_set_root(out, obj);
        yyjson_mut_obj_add_strcpy(out, obj, "id", job->id.c_str());
        res.set_content(json_string(out), "application/json");
        yyjson_mut_doc_free(out);
    });

    svr.Post("/synth", [&engine, &parse_json_object](const httplib::Request& req, httplib::Response& res) {
        yyjson_doc* doc = nullptr;
        yyjson_val* root = parse_json_object(req, doc);
        if (!root) {
            send_json_error(res, 400, "invalid json");
            return;
        }
        std::string queue_err;
        auto job = engine.submit_synth(root, queue_err);
        yyjson_doc_free(doc);
        if (!job) {
            send_json_error(res, 429, queue_err);
            return;
        }
        yyjson_mut_doc* out = yyjson_mut_doc_new(nullptr);
        yyjson_mut_val* obj = yyjson_mut_obj(out);
        yyjson_mut_doc_set_root(out, obj);
        yyjson_mut_obj_add_strcpy(out, obj, "id", job->id.c_str());
        res.set_content(json_string(out), "application/json");
        yyjson_mut_doc_free(out);
    });

    svr.Post("/job", [&engine](const httplib::Request& req, httplib::Response& res) {
        const std::string id = req.get_param_value("id");
        if (req.has_param("cancel") && !req.get_param_value("cancel").empty()) {
            if (!engine.cancel_job(id)) {
                send_json_error(res, 404, "unknown job");
                return;
            }
            res.set_content("{\"cancelled\":true}", "application/json");
            return;
        }
        send_json_error(res, 404, "not found");
    });

    std::signal(SIGTERM, handle_signal);
    std::signal(SIGINT, handle_signal);

    std::atomic<bool> server_exited{false};
    std::thread stopper([&] {
        while (!server_exited.load(std::memory_order_relaxed)) {
            if (g_stop_requested.load(std::memory_order_relaxed)) {
                engine.shutdown();
                svr.stop();
                return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    });

    const bool ok = svr.listen(host, port);
    server_exited.store(true, std::memory_order_relaxed);
    if (stopper.joinable()) stopper.join();
    engine.shutdown();
    return (ok || g_stop_requested.load(std::memory_order_relaxed)) ? 0 : 1;
}
