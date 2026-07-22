#pragma once

#include <atomic>
#include <expected>
#include <memory>
#include <string>
#include <vector>

#include <curl/curl.h>
#include <nlohmann/json.hpp>

#include "llmclient/types.h"

namespace llmclient {

// HTTP client for OpenAI-compatible Chat Completions APIs.
class Client {
  public:
    Client(std::string api_base, std::string api_key = {});
    ~Client();

    Client(const Client&) = delete;
    Client& operator=(const Client&) = delete;
    Client(Client&&) = delete;
    Client& operator=(Client&&) = delete;

    // Cancellation token (shared_ptr<atomic<bool>>)
    using Token = std::shared_ptr<std::atomic<bool>>;
    void set_cancelled(Token t) { cancelled_ = std::move(t); }

    void set_api_base(const std::string& base) { api_base_ = base; }
    void set_api_key(const std::string& key) { api_key_ = key; }

    // Non-streaming chat completion
    std::expected<nlohmann::json, std::string> chat(const nlohmann::json& payload);

    // Streaming chat completion
    std::expected<void, std::string> stream_chat(const nlohmann::json& payload, SSEParser::Callbacks callbacks);

    // Query the API for model metadata, returning the context window size if found.
    // Returns 0 if the endpoint doesn't expose it (caller should use a default).
    int fetch_model_context_limit(const std::string& model);

    // Query /v1/models and return the list of model IDs (empty on error).
    std::expected<std::vector<std::string>, std::string> fetch_models();

    const std::string& last_raw_response() const { return raw_response_; }
    std::string url() const {
        return api_base_ + "/chat/completions";
    }
    std::string models_url() const { return api_base_ + "/models"; }

  private:
    static constexpr int kMaxRetries = 3;
    static constexpr double kBaseDelaySec = 1.0;

    struct curl_slist* make_headers() const;
    bool should_retry(long http_code) const;
    CURLcode perform_with_retry(CURL* curl, long& http_code, std::string& body);

    // Low-level HTTP GET helper
    std::expected<std::string, std::string> http_get(const std::string& url);

    Token cancelled_;
    std::string api_base_;
    std::string api_key_;
    std::string raw_response_;

    static size_t write_body(char* ptr, size_t size, size_t nmemb, void* userdata);
    static size_t write_stream(char* ptr, size_t size, size_t nmemb, void* userdata);
};

} // namespace llmclient
