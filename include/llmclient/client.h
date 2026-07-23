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

    // ── Payload builders ──

    /// Build a chat completion request payload.
    /// Handles model, messages, tools, stream flag, max_tokens/max_completion_tokens,
    /// stream_options, and thinking/reasoning parameters.
    ///
    /// @param tools  JSON array of tool definitions, or `nullptr`/empty array to
    ///               indicate no tools.  Non-array, non-null values throw
    ///               `std::invalid_argument`.  When null or empty, the `tools` key
    ///               is omitted from the request payload.
    /// @throws std::invalid_argument  if tools is not a JSON array and not null
    nlohmann::json build_chat_request(const std::string& model,
                                      const nlohmann::json& messages,
                                      const nlohmann::json& tools,
                                      bool stream,
                                      int max_tokens_hint = 0,
                                      int context_limit = 0,
                                      bool thinking_enabled = false) const;

    /// Returns true if the model name matches known reasoning/thinking model patterns.
    static bool model_supports_thinking(const std::string& model);

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

    /// Apply thinking/reasoning parameters to a payload.
    void apply_thinking_params(nlohmann::json& payload, int limit, const std::string& model, bool thinking_enabled) const;

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
