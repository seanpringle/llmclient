#include "llmclient/client.h"

#include <cstring>
#include <iostream>
#include <random>
#include <thread>
#include <chrono>

llmclient::Client::Client(std::string api_base, std::string api_key)
    : api_base_(std::move(api_base)), api_key_(std::move(api_key)) {
    while (!api_base_.empty() && api_base_.back() == '/') {
        api_base_.pop_back();
    }
}

llmclient::Client::~Client() = default;

struct curl_slist* llmclient::Client::make_headers() const {
    struct curl_slist* list = nullptr;
    list = curl_slist_append(list, "Content-Type: application/json");
    if (!api_key_.empty()) {
        std::string auth_header = "Authorization: Bearer " + api_key_;
        list = curl_slist_append(list, auth_header.c_str());
    }
    return list;
}

size_t llmclient::Client::write_body(char* ptr, size_t size, size_t nmemb, void* userdata) {
    auto* s = static_cast<std::string*>(userdata);
    s->append(ptr, size * nmemb);
    return size * nmemb;
}

size_t llmclient::Client::write_stream(char* ptr, size_t size, size_t nmemb, void* userdata) {
    auto* parser = static_cast<SSEParser*>(userdata);
    parser->feed(ptr, size * nmemb);
    return size * nmemb;
}

static int progress_cb(void* clientp, curl_off_t /*dltotal*/, curl_off_t /*dlnow*/, curl_off_t /*ultotal*/, curl_off_t /*ulnow*/) {
    auto* cancelled = static_cast<std::atomic<bool>*>(clientp);
    return (cancelled && *cancelled) ? 1 : 0;
}

static CURL* setup_curl(const std::string& url, struct curl_slist* headers, const std::string& payload_str, std::atomic<bool>* cancelled = nullptr) {
    CURL* curl = curl_easy_init();
    if (!curl)
        return nullptr;

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payload_str.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, static_cast<long>(payload_str.size()));

    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 10L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 3600L);

    // NOSIGNAL prevents curl from raising SIGPIPE on e.g. dropped connections.
    curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);
    // Keep-alive to detect half-open TCP connections during long streaming reads.
    curl_easy_setopt(curl, CURLOPT_TCP_KEEPALIVE, 1L);
    curl_easy_setopt(curl, CURLOPT_TCP_KEEPIDLE, 60L);
    curl_easy_setopt(curl, CURLOPT_TCP_KEEPINTVL, 30L);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "cima/1.0");
    // Try HTTP/2 for multiplexed streaming; falls back to HTTP/1.1 automatically.
    curl_easy_setopt(curl, CURLOPT_HTTP_VERSION, CURL_HTTP_VERSION_2TLS);

    // Enable SSL/TLS verification when using HTTPS.
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 1L);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 2L);
    // Use system default CA bundle — do NOT set a custom CA path so that
    // curl finds the system trust store automatically.
    curl_easy_setopt(curl, CURLOPT_CAINFO, nullptr);

    // Enable transparent decompression (gzip, deflate, etc.).
    // libcurl will send Accept-Encoding and auto-decompress the response.
    curl_easy_setopt(curl, CURLOPT_ACCEPT_ENCODING, "");

    curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, progress_cb);
    curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);
    curl_easy_setopt(curl, CURLOPT_XFERINFODATA, cancelled);

    return curl;
}

// Retry on rate-limit (429) or server errors (5xx) — but not 4xx client errors
// which indicate a malformed request that will keep failing.
bool llmclient::Client::should_retry(long http_code) const {
    return http_code == 429 || (http_code >= 500 && http_code < 600);
}

/// Returns a random delay in [0.5*base, 1.5*base] to add jitter to retries.
static double jittered_delay(double base_sec) {
    // thread_local so we seed once per thread (fine for single-threaded UI)
    static thread_local std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<double> dist(0.5, 1.5);
    return base_sec * dist(rng);
}

CURLcode llmclient::Client::perform_with_retry(CURL* curl, long& http_code, std::string& body) {
    for (int attempt = 0; attempt < kMaxRetries; ++attempt) {
        body.clear();
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_body);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &body);

        CURLcode res = curl_easy_perform(curl);
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

        if (res == CURLE_OK && !should_retry(http_code))
            return CURLE_OK;

        if (attempt == kMaxRetries - 1)
            return res;

        // Recoverable: HTTP-level errors on successful connections (e.g. 5xx, 429)
        // plus transient transport failures that may succeed on retry.
        bool recoverable = (res == CURLE_OK && should_retry(http_code)) || res == CURLE_SEND_ERROR || res == CURLE_RECV_ERROR ||
            res == CURLE_OPERATION_TIMEDOUT || res == CURLE_COULDNT_CONNECT;
        if (!recoverable)
            return res;

        std::this_thread::sleep_for(std::chrono::duration<double>(jittered_delay(kBaseDelaySec * (1 << attempt))));
    }
    return CURLE_OK;
}

// ── Low-level HTTP GET ──
std::expected<std::string, std::string> llmclient::Client::http_get(const std::string& url) {
    auto* headers = make_headers();
    CURL* curl = curl_easy_init();
    if (!curl) {
        curl_slist_free_all(headers);
        return std::unexpected(std::string("curl_easy_init failed"));
    }

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_HTTPGET, 1L);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 10L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 30L);
    curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "cima/1.0");
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 1L);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 2L);
    curl_easy_setopt(curl, CURLOPT_CAINFO, nullptr);

    curl_easy_setopt(curl, CURLOPT_ACCEPT_ENCODING, "");

    curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, progress_cb);
    curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);
    curl_easy_setopt(curl, CURLOPT_XFERINFODATA, cancelled_.get());

    std::string body;
    long http_code = 0;
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_body);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &body);

    CURLcode res = perform_with_retry(curl, http_code, body);
    curl_easy_cleanup(curl);
    curl_slist_free_all(headers);

    if (res != CURLE_OK) {
        return std::unexpected(std::string("curl error: ") + curl_easy_strerror(res));
    }
    if (http_code != 200) {
        // Log request URL and error response for 4xx errors
        if (http_code >= 400 && http_code < 500) {
            std::cerr << "HTTP " << http_code << " error for GET " << url << "\n";
            if (!body.empty()) {
                std::cerr << "Response body:\n" << body << std::endl;
            }
        }
        return std::unexpected("HTTP " + std::to_string(http_code));
    }
    return body;
}

// ── Discover context window from API metadata ──
int llmclient::Client::fetch_model_context_limit(const std::string& model) {
    // Process-wide cache: all sessions share discovered limits
    // (keyed by URL + model).
    static std::mutex cache_mutex;
    static std::unordered_map<std::string, int> cache;

    std::string cache_key = api_base_ + ":" + model;
    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        auto it = cache.find(cache_key);
        if (it != cache.end()) {
            return it->second;
        }
    }

    auto body = http_get(models_url());
    if (!body) {
        return 0;
    }

    nlohmann::json j;
    try {
        j = nlohmann::json::parse(*body);
    } catch (...) {
        return 0;
    }

    nlohmann::json model_obj;

    // Try "data" array first (OpenAI-style: {"object":"list","data":[{...}]})
    if (j.is_object() && j.contains("data") && j["data"].is_array()) {
        for (const auto& entry : j["data"]) {
            if (entry.value("id", "") == model || entry.value("name", "") == model) {
                model_obj = entry;
                break;
            }
        }
        // If we didn't find the specific model, try the first one
        if (model_obj.is_null() && !j["data"].empty()) {
            model_obj = j["data"][0];
        }
    } else if (j.is_object() && j.value("id", "") == model) {
        // Single model object
        model_obj = j;
    }

    if (model_obj.is_null()) {
        return 0;
    }

    // Known field names across backends, in priority order
    static const char* const kContextFields[] = {
        "context_window",
        "max_model_len",
        "max_context_length",
        "context_length",
        "inputTokenLimit",
        "max_input_tokens",
        "max_total_tokens",
    };

    for (const auto& field : kContextFields) {
        auto it = model_obj.find(field);
        if (it != model_obj.end() && it->is_number_integer()) {
            int val = it->get<int>();
            if (val > 0) {
                // Cache the result before returning.
                std::lock_guard<std::mutex> lock(cache_mutex);
                cache[cache_key] = val;
                return val;
            }
        }
    }

    // Model found but no context window field. Cache a zero so we don't
    // re-query on every request — the caller will use its own default.
    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        cache[cache_key] = 0;
    }
    return 0;
}

// ── Fetch available models from /v1/models ──
std::expected<std::vector<std::string>, std::string> llmclient::Client::fetch_models() {
    auto body = http_get(models_url());
    if (!body) {
        return std::unexpected(body.error());
    }

    nlohmann::json j;
    try {
        j = nlohmann::json::parse(*body);
    } catch (const nlohmann::json::parse_error& e) {
        return std::unexpected(std::string("JSON parse error: ") + e.what());
    }

    std::vector<std::string> models;
    if (j.is_object() && j.contains("data") && j["data"].is_array()) {
        for (const auto& entry : j["data"]) {
            if (entry.is_object() && entry.contains("id") && entry["id"].is_string()) {
                models.push_back(entry["id"].get<std::string>());
            }
        }
    }

    return models;
}

std::expected<nlohmann::json, std::string> llmclient::Client::chat(const nlohmann::json& payload) {
    std::string payload_str = payload.dump();
    std::string body;
    long http_code = 0;

    auto* headers = make_headers();
    CURL* curl = setup_curl(url(), headers, payload_str, cancelled_.get());
    if (!curl) {
        curl_slist_free_all(headers);
        return std::unexpected(std::string("curl_easy_init failed"));
    }

    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_body);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &body);

    CURLcode res = perform_with_retry(curl, http_code, body);
    curl_easy_cleanup(curl);
    curl_slist_free_all(headers);

    raw_response_ = body;

    if (res != CURLE_OK) {
        auto msg = std::string("curl error: ") + curl_easy_strerror(res);
        if (http_code != 0) {
            msg += " (HTTP " + std::to_string(http_code) + ")";
        }
        return std::unexpected(std::move(msg));
    }

    if (http_code != 200) {
        // Log request details and error response for 4xx errors
        if (http_code >= 400 && http_code < 500) {
            std::cerr << "HTTP " << http_code << " error for POST " << url() << "\n";
            std::cerr << "Request body:\n" << payload_str << "\n";
            if (!body.empty()) {
                std::cerr << "Response body:\n" << body << std::endl;
            }
        } else {
            // Log just the response body for other errors
            if (!body.empty()) {
                std::cerr << "HTTP " << http_code << " response body:\n" << body << std::endl;
            }
        }
        std::string msg = "HTTP " + std::to_string(http_code);
        if (!body.empty()) {
            msg += ": " + body.substr(0, 500);
        }
        return std::unexpected(msg);
    }

    try {
        return nlohmann::json::parse(body);
    } catch (const nlohmann::json::parse_error& e) {
        return std::unexpected("JSON parse error: " + std::string(e.what()));
    }
}

std::expected<void, std::string> llmclient::Client::stream_chat(const nlohmann::json& payload, SSEParser::Callbacks callbacks) {
    std::string payload_str = payload.dump();
    long http_code = 0;

    // Wrap callbacks to detect if any data was already delivered to the user.
    // We must not retry if callbacks were invoked — retrying would duplicate data.
    bool data_delivered = false;
    SSEParser::Callbacks guarded;
    if (callbacks.on_data) {
        guarded.on_data = [&data_delivered, cb = std::move(callbacks.on_data)](const std::string& ev, const nlohmann::json& j) {
            data_delivered = true;
            cb(ev, j);
        };
    }
    if (callbacks.on_done) {
        guarded.on_done = [&data_delivered, cb = std::move(callbacks.on_done)]() {
            data_delivered = true;
            cb();
        };
    }
    if (callbacks.on_error) {
        guarded.on_error = [&data_delivered, cb = std::move(callbacks.on_error)](const std::string& s) {
            data_delivered = true;
            cb(s);
        };
    }
    // ── Forward structured streaming callbacks ──
    if (callbacks.on_content_delta) {
        guarded.on_content_delta = [&data_delivered, cb = std::move(callbacks.on_content_delta)](std::string_view t) {
            data_delivered = true;
            cb(t);
        };
    }
    if (callbacks.on_reasoning_delta) {
        guarded.on_reasoning_delta = [&data_delivered, cb = std::move(callbacks.on_reasoning_delta)](std::string_view t) {
            data_delivered = true;
            cb(t);
        };
    }
    if (callbacks.on_tool_call_delta) {
        guarded.on_tool_call_delta = [&data_delivered, cb = std::move(callbacks.on_tool_call_delta)](const nlohmann::json& d) {
            data_delivered = true;
            cb(d);
        };
    }
    if (callbacks.on_usage) {
        guarded.on_usage = [&data_delivered, cb = std::move(callbacks.on_usage)](Usage u) {
            data_delivered = true;
            cb(u);
        };
    }

    SSEParser parser(std::move(guarded));

    auto* headers = make_headers();
    CURL* curl = setup_curl(url(), headers, payload_str, cancelled_.get());
    if (!curl) {
        curl_slist_free_all(headers);
        return std::unexpected(std::string("curl_easy_init failed"));
    }

    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_stream);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &parser);

    CURLcode res = CURLE_OK;
    for (int attempt = 0; attempt < kMaxRetries; ++attempt) {
        data_delivered = false;
        parser.reset();
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_stream);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &parser);

        res = curl_easy_perform(curl);
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

        if (res == CURLE_OK && !should_retry(http_code))
            break;

        if (attempt == kMaxRetries - 1)
            break;

        // If any callback was already invoked, data was consumed by the user.
        // Retrying would produce duplicate content — bail out instead.
        if (data_delivered)
            break;

        bool recoverable = (res == CURLE_OK && should_retry(http_code)) || res == CURLE_SEND_ERROR || res == CURLE_RECV_ERROR ||
            res == CURLE_OPERATION_TIMEDOUT || res == CURLE_COULDNT_CONNECT;
        if (!recoverable)
            break;

        std::this_thread::sleep_for(std::chrono::duration<double>(jittered_delay(kBaseDelaySec * (1 << attempt))));
    }

    parser.flush();
    curl_easy_cleanup(curl);
    curl_slist_free_all(headers);

    raw_response_ = parser.raw();

    if (res != CURLE_OK) {
        // Log the full raw response to stderr for debugging.
        if (!raw_response_.empty()) {
            std::cerr << "curl error raw response (" << curl_easy_strerror(res) << "):\n" << raw_response_ << std::endl;
        }
        auto msg = std::string("curl error: ") + curl_easy_strerror(res);
        if (!raw_response_.empty()) {
            msg += " | raw: " + raw_response_.substr(0, 500);
        }
        return std::unexpected(std::move(msg));
    }

    if (http_code != 200) {
        // Log request details and error response for 4xx errors
        if (http_code >= 400 && http_code < 500) {
            std::cerr << "HTTP " << http_code << " error for POST " << url() << "\n";
            std::cerr << "Request body:\n" << payload_str << "\n";
            if (!raw_response_.empty()) {
                std::cerr << "Response body:\n" << raw_response_ << std::endl;
            }
        } else {
            // Log just the response body for other errors
            if (!raw_response_.empty()) {
                std::cerr << "HTTP " << http_code << " response body:\n" << raw_response_ << std::endl;
            }
        }
        auto msg = "HTTP " + std::to_string(http_code);
        if (!raw_response_.empty()) {
            msg += ": " + raw_response_.substr(0, 500);
        }
        return std::unexpected(msg);
    }

    return {};
}

// ── Payload building ──

nlohmann::json llmclient::Client::build_chat_request(const std::string& model,
                                                     const nlohmann::json& messages,
                                                     const nlohmann::json& tools,
                                                     bool stream,
                                                     int max_tokens_hint,
                                                     int context_limit,
                                                     bool thinking_enabled) const {
    nlohmann::json payload = {
        {"model", model},
        {"messages", messages},
        {"tools", tools},
        {"stream", stream},
        {"stream_options", {{"include_usage", true}}}
    };

    int limit = max_tokens_hint > 0 ? max_tokens_hint
        : (context_limit > 0 ? std::min(context_limit / 4, 32768) : 32768);
    apply_thinking_params(payload, limit, model, thinking_enabled);
    return payload;
}

void llmclient::Client::apply_thinking_params(nlohmann::json& payload, int limit,
                                              const std::string& model,
                                              bool thinking_enabled) const {
    if (thinking_enabled) {
        std::string lower = model;
        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
        bool is_openai = lower.find("o1") != std::string::npos
                      || lower.find("o3") != std::string::npos
                      || lower.find("o4") != std::string::npos
                      || lower.find("gpt-5") != std::string::npos;
        if (is_openai) {
            payload["reasoning_effort"] = "high";
        } else {
            payload["thinking"] = {{"type", "enabled"}};
        }
        payload["max_completion_tokens"] = limit;
    } else {
        payload["max_tokens"] = limit;
    }
}

bool llmclient::Client::model_supports_thinking(const std::string& model) {
    static const char* keywords[] = {
        "deepseek",  // DeepSeek V4 Pro/Flash, R1, V3
        "mimo",      // Xiaomi MiMo V2.5, V2.5-Pro, V2-Pro, V2-Omni
        "glm",       // Zhipu GLM-5, GLM-5.1, GLM-5.2
        "qwen3",     // Alibaba Qwen 3.x via llama.cpp thinking mode
        "o1",        // OpenAI o1, o1-mini, o1-pro
        "o3",        // OpenAI o3, o3-mini, o3-pro
        "o4",        // OpenAI o4-mini
        "gpt-5",     // OpenAI GPT-5, GPT-5.1, GPT-5.2, GPT-5.4, GPT-5.5
    };
    std::string lower = model;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    for (const auto* kw : keywords) {
        if (lower.find(kw) != std::string::npos)
            return true;
    }
    return false;
}
