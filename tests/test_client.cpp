#include <llmclient/client.h>
#include "mock_server.hpp"

#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <memory>
#include <thread>

using json = nlohmann::json;
using namespace llmclient;

// ===================================================================
// Non-streaming chat
// ===================================================================

TEST_CASE("Client non-streaming returns JSON", "[client]") {
    MockServer server([](const std::string&) -> std::string {
        return R"({"id":"test-1","choices":[{"message":{"role":"assistant","content":"Hello!"}}]})";
    });

    Client client(server.base_url());
    json payload = {{"model", "test"},
                    {"messages",
                     {{{"role", "user"}, {"content", "say hi"}}}}};

    auto result = client.chat(payload);
    REQUIRE(result);
    CHECK((*result)["id"] == "test-1");
    CHECK((*result)["choices"][0]["message"]["content"] == "Hello!");
}

TEST_CASE("Client non-streaming errors on bad URL", "[client]") {
    Client client("http://127.0.0.1:1/v1");
    json payload = {{"model", "test"}, {"messages", {}}};

    auto result = client.chat(payload);
    CHECK_FALSE(result);
    CHECK((result.error().find("curl error") != std::string::npos ||
           result.error().find("HTTP") != std::string::npos));
}

TEST_CASE("Client non-streaming sends auth header", "[client]") {
    bool auth_seen = false;
    MockServer server([&](const std::string& req) -> std::string {
        if (req.find("Authorization: Bearer sk-test123") != std::string::npos) {
            auth_seen = true;
        }
        return R"({"id":"auth-test","choices":[{"message":{"role":"assistant","content":"ok"}}]})";
    });

    Client client(server.base_url(), "sk-test123");
    json payload = {{"model", "test"}, {"messages", {}}};
    auto result = client.chat(payload);
    REQUIRE(result);
    CHECK(auth_seen);
}

TEST_CASE("Client non-streaming no auth when key empty", "[client]") {
    bool auth_seen = false;
    MockServer server([&](const std::string& req) -> std::string {
        if (req.find("Authorization:") != std::string::npos) {
            auth_seen = true;
        }
        return R"({"id":"noauth","choices":[{"message":{"role":"assistant","content":"ok"}}]})";
    });

    Client client(server.base_url());
    json payload = {{"model", "test"}, {"messages", {}}};
    auto result = client.chat(payload);
    REQUIRE(result);
    CHECK_FALSE(auth_seen);
}

// ===================================================================
// Streaming chat
// ===================================================================

TEST_CASE("Client streaming calls on_data and on_done", "[client]") {
    MockServer server(
        [](const std::string&) -> std::string {
            return "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\ndata: {\"choices\":[{\"delta\":{\"content\":\" world\"}}]}\n\ndata: [DONE]\n\n";
        },
        true);

    Client client(server.base_url());
    json payload = {{"model", "test"},
                    {"messages",
                     {{{"role", "user"}, {"content", "hi"}}}}};

    std::string content;
    bool done = false;
    bool errored = false;

    SSEParser::Callbacks cbs;
    cbs.on_data = [&](const std::string&, const json& j) {
        content += j["choices"][0]["delta"].value("content", "");
    };
    cbs.on_done = [&]() { done = true; };
    cbs.on_error = [&](const std::string&) { errored = true; };

    auto result = client.stream_chat(payload, std::move(cbs));
    REQUIRE(result);
    CHECK(content == "Hello world");
    CHECK(done);
    CHECK_FALSE(errored);
}

TEST_CASE("Client streaming calls on_error for bad JSON", "[client]") {
    MockServer server(
        [](const std::string&) -> std::string {
            // Contains invalid SSE data
            return "data: {invalid}\n\ndata: [DONE]\n\n";
        },
        true);

    Client client(server.base_url());
    json payload = {{"model", "test"}, {"messages", {}}};

    bool errored = false;
    SSEParser::Callbacks cbs;
    cbs.on_data = [](const std::string&, const json&) {};
    cbs.on_done = []() {};
    cbs.on_error = [&](const std::string&) { errored = true; };

    auto result = client.stream_chat(payload, std::move(cbs));
    // The transfer should succeed (HTTP 200), but the parse error fires
    CHECK(result);
    CHECK(errored);
}

// ===================================================================
// Structured streaming callbacks through Client
// ===================================================================

TEST_CASE("Client streaming structured callbacks fire correctly", "[client]") {
    // SSE with content, reasoning, tool calls, usage, and [DONE]
    MockServer server(
        [](const std::string&) -> std::string {
            return "data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"reasoning_content\":\"Thinking...\",\"content\":\"Hello\"}}]}\n\n"
                   "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\" world\"}}]}\n\n"
                   "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"function\":{\"name\":\"test\",\"arguments\":\"{}\"}}]}}]}\n\n"
                   "data: {\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":20,\"total_tokens\":30}}\n\n"
                   "data: [DONE]\n\n";
        },
        true);

    Client client(server.base_url());
    json payload = {{"model", "test"}, {"messages", {}}};

    std::string content;
    std::string reasoning;
    int tool_call_count = 0;
    ToolAccumulator tool_acc;
    bool usage_seen = false;
    bool done = false;
    bool errored = false;
    int data_cb_count = 0;

    SSEParser::Callbacks cbs;
    cbs.on_data = [&](const std::string&, const json&) { data_cb_count++; };
    cbs.on_done = [&]() { done = true; };
    cbs.on_error = [&](const std::string&) { errored = true; };
    cbs.on_content_delta = [&](std::string_view t) { content += std::string(t); };
    cbs.on_reasoning_delta = [&](std::string_view t) { reasoning += std::string(t); };
    cbs.on_tool_call_delta = [&](const json& d) { tool_call_count++; tool_acc.apply(d); };
    cbs.on_usage = [&](Usage u) {
        usage_seen = true;
        CHECK(u.prompt_tokens == 10);
        CHECK(u.completion_tokens == 20);
        CHECK(u.total_tokens == 30);
    };

    auto result = client.stream_chat(payload, std::move(cbs));
    REQUIRE(result);
    CHECK_FALSE(errored);
    CHECK(done);
    CHECK(content == "Hello world");
    CHECK(reasoning == "Thinking...");
    CHECK(tool_call_count == 1);
    auto calls = tool_acc.finalize();
    REQUIRE(calls.size() == 1);
    CHECK(calls[0].id == "call_1");
    CHECK(calls[0].name == "test");
    CHECK(usage_seen);
    // on_data should have been called for every event (4 data events + 1 [DONE] = 5, but [DONE] doesn't call on_data, so 4)
    CHECK(data_cb_count == 4);
}

// ===================================================================
// last_raw_response
// ===================================================================

TEST_CASE("Client last_raw_response after chat", "[client]") {
    MockServer server([](const std::string&) -> std::string {
        return R"({"id":"test-raw","choices":[{"message":{"role":"assistant","content":"raw"}}]})";
    });

    Client client(server.base_url());
    json payload = {{"model", "test"}, {"messages", {}}};
    auto result = client.chat(payload);
    REQUIRE(result);
    CHECK(client.last_raw_response().find("test-raw") != std::string::npos);
}

// ===================================================================
// fetch_models
// ===================================================================

TEST_CASE("Client fetch_models returns parsed model IDs", "[client]") {
    MockServer server([](const std::string& req) -> std::string {
        // Check it's hitting the models endpoint
        if (req.find("/v1/models") != std::string::npos) {
            return R"({"object":"list","data":[
                {"id":"model-a","object":"model"},
                {"id":"model-b","object":"model"}
            ]})";
        }
        return R"({"error":"unexpected"})";
    });

    Client client(server.base_url());
    auto models = client.fetch_models();
    REQUIRE(models);
    REQUIRE(models->size() == 2);
    CHECK((*models)[0] == "model-a");
    CHECK((*models)[1] == "model-b");
}

TEST_CASE("Client fetch_models returns error on bad URL", "[client]") {
    Client client("http://127.0.0.1:1/v1");
    auto models = client.fetch_models();
    CHECK_FALSE(models);
}

// ===================================================================
// fetch_model_context_limit
// ===================================================================

TEST_CASE("Client fetch_model_context_limit from /v1/models", "[client]") {
    MockServer server([](const std::string& req) -> std::string {
        if (req.find("/v1/models") != std::string::npos) {
            return R"({"object":"list","data":[
                {"id":"test-model","object":"model","context_window":32000}
            ]})";
        }
        return R"({"error":"unexpected"})";
    });

    Client client(server.base_url());
    int limit = client.fetch_model_context_limit("test-model");
    CHECK(limit == 32000);
}

TEST_CASE("Client fetch_model_context_limit fallback model with 0 context_window", "[client]") {
    // When the requested model isn't in the list, the function falls back
    // to the first model in the data array. If that fallback has a
    // context_window of 0 (or no usable field), the result (0) is cached
    // so we don't re-query on every call.
    MockServer server([](const std::string& req) -> std::string {
        if (req.find("/v1/models") != std::string::npos) {
            return R"({"object":"list","data":[
                {"id":"other-model","object":"model","context_window":0}
            ]})";
        }
        return R"({"error":"unexpected"})";
    });

    Client client(server.base_url());
    int limit = client.fetch_model_context_limit("nonexistent");
    CHECK(limit == 0);
}

TEST_CASE("Client fetch_model_context_limit bad URL returns 0", "[client]") {
    Client client("http://127.0.0.1:1/v1");
    int limit = client.fetch_model_context_limit("test");
    CHECK(limit == 0);
}

// ===================================================================
// fetch_model_context_limit caching
// ===================================================================

TEST_CASE("Client fetch_model_context_limit caches second call", "[client]") {
    std::atomic<int> call_count{0};
    MockServer server([&](const std::string& req) -> std::string {
        call_count++;
        if (req.find("/v1/models") != std::string::npos) {
            return R"({"object":"list","data":[
                {"id":"test-model","object":"model","context_window":16000}
            ]})";
        }
        return R"({"error":"unexpected"})";
    });

    Client client(server.base_url());

    // First call — should hit the server
    int first = client.fetch_model_context_limit("test-model");
    CHECK(first == 16000);
    CHECK(call_count == 1);

    // Second call with same model — should use cache
    int second = client.fetch_model_context_limit("test-model");
    CHECK(second == 16000);
    CHECK(call_count == 1); // no additional HTTP request
}

TEST_CASE("Client fetch_model_context_limit separate cache entries per model", "[client]") {
    std::atomic<int> call_count{0};
    MockServer server([&](const std::string& req) -> std::string {
        call_count++;
        if (req.find("/v1/models") != std::string::npos) {
            return R"({"object":"list","data":[
                {"id":"model-a","object":"model","context_window":8000},
                {"id":"model-b","object":"model","context_window":32000}
            ]})";
        }
        return R"({"error":"unexpected"})";
    });

    Client client(server.base_url());

    int a = client.fetch_model_context_limit("model-a");
    CHECK(a == 8000);
    CHECK(call_count == 1);

    int b = client.fetch_model_context_limit("model-b");
    CHECK(b == 32000);
    CHECK(call_count == 2); // different model, new request
}

TEST_CASE("Client fetch_model_context_limit separate cache per base URL", "[client]") {
    int call_count_a = 0, call_count_b = 0;
    MockServer server_a([&](const std::string& req) -> std::string {
        call_count_a++;
        if (req.find("/v1/models") != std::string::npos)
            return R"({"object":"list","data":[{"id":"m","object":"model","context_window":8000}]})";
        return R"({"error":"unexpected"})";
    });
    MockServer server_b([&](const std::string& req) -> std::string {
        call_count_b++;
        if (req.find("/v1/models") != std::string::npos)
            return R"({"object":"list","data":[{"id":"m","object":"model","context_window":16000}]})";
        return R"({"error":"unexpected"})";
    });

    Client client_a(server_a.base_url());
    Client client_b(server_b.base_url());

    CHECK(client_a.fetch_model_context_limit("m") == 8000);
    CHECK(call_count_a == 1);

    CHECK(client_b.fetch_model_context_limit("m") == 16000);
    CHECK(call_count_b == 1);

    // Calling client_a again with same model should return cached value
    CHECK(client_a.fetch_model_context_limit("m") == 8000);
    CHECK(call_count_a == 1); // no new request to server_a
}

TEST_CASE("Client fetch_model_context_limit does not cache HTTP errors", "[client]") {
    std::atomic<int> call_count{0};
    // Mock that always returns something that isn't a valid models response.
    // First call: fails → returns 0, not cached.
    // Second call: should retry (not served from cache).
    MockServer server([&](const std::string&) -> std::string {
        call_count++;
        return R"({"error":"not a models list"})";
    });
    Client client(server.base_url());

    CHECK(client.fetch_model_context_limit("m") == 0);
    CHECK(call_count == 1);

    // Second call — should NOT be cached (HTTP error wasn't cached)
    CHECK(client.fetch_model_context_limit("m") == 0);
    CHECK(call_count == 2);
}

// ===================================================================
// Retry behavior
// ===================================================================

TEST_CASE("Client retries on 429 Too Many Requests", "[client]") {
    std::atomic<int> call_count{0};
    MockServer server(
        [&](const std::string&) -> std::string {
            call_count++;
            return R"({"error":"rate limited"})";
        },
        false, 429);

    Client client(server.base_url());
    json payload = {{"model", "test"}, {"messages", {}}};
    auto result = client.chat(payload);

    // Should fail eventually (after retries), but call_count > 1 means retries happened.
    CHECK_FALSE(result);
    CHECK(call_count > 1);
}

TEST_CASE("Client retries on 503 Service Unavailable", "[client]") {
    std::atomic<int> call_count{0};
    MockServer server(
        [&](const std::string&) -> std::string {
            call_count++;
            return R"({"error":"unavailable"})";
        },
        false, 503);

    Client client(server.base_url());
    json payload = {{"model", "test"}, {"messages", {}}};
    auto result = client.chat(payload);

    CHECK_FALSE(result);
    CHECK(call_count > 1);
}

TEST_CASE("Client does NOT retry on 400 Bad Request", "[client]") {
    std::atomic<int> call_count{0};
    MockServer server(
        [&](const std::string&) -> std::string {
            call_count++;
            return R"({"error":"bad request"})";
        },
        false, 400);

    Client client(server.base_url());
    json payload = {{"model", "test"}, {"messages", {}}};
    auto result = client.chat(payload);

    CHECK_FALSE(result);
    CHECK(call_count == 1);  // no retry
}

// ===================================================================
// Cancellation
// ===================================================================

TEST_CASE("Client cancellation during streaming stops mid-stream", "[client]") {
    std::atomic<int> call_count{0};
    auto cancelled = std::make_shared<std::atomic<bool>>(false);

    MockServer server(
        [&](const std::string&) -> std::string {
            call_count++;
            // Return a streaming response that sends one chunk then pauses
            return "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\n";
        },
        true);

    Client client(server.base_url());
    client.set_cancelled(cancelled);

    json payload = {{"model", "test"}, {"messages", {}}};

    std::string content;
    SSEParser::Callbacks cbs;
    cbs.on_data = [&](const std::string&, const json& j) {
        content += j["choices"][0]["delta"].value("content", "");
        // Cancel after receiving first data
        *cancelled = true;
    };
    cbs.on_done = []() {};
    cbs.on_error = [](const std::string&) {};

    // The stream might return an error due to cancellation, or succeed
    // but with cancelled token. Either is acceptable.
    auto result = client.stream_chat(payload, std::move(cbs));
    // We should have received some content
    CHECK(content == "Hello");
}

// ===================================================================
// Streaming with tool call deltas
// ===================================================================

TEST_CASE("Client streaming with tool call delta", "[client]") {
    MockServer server(
        [](const std::string&) -> std::string {
            return "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"function\":{\"name\":\"read_file\",\"arguments\":\"\"}}]}}]}\n\n"
                   "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"path\\\":\\\"test.txt\\\"}\"}}]}}]}\n\n"
                   "data: [DONE]\n\n";
        },
        true);

    Client client(server.base_url());
    json payload = {{"model", "test"}, {"messages", {}}};

    ToolAccumulator acc;
    bool done = false;

    SSEParser::Callbacks cbs;
    cbs.on_data = [&](const std::string&, const json& j) {
        auto choices = j.find("choices");
        if (choices != j.end() && choices->is_array() && !choices->empty()) {
            const auto& delta = (*choices)[0]["delta"];
            acc.apply(delta);
        }
    };
    cbs.on_done = [&]() { done = true; };
    cbs.on_error = [](const std::string&) {};

    auto result = client.stream_chat(payload, std::move(cbs));
    REQUIRE(result);

    CHECK(done);
    auto calls = acc.finalize();
    REQUIRE(calls.size() == 1);
    CHECK(calls[0].id == "call_1");
    CHECK(calls[0].name == "read_file");
    CHECK(calls[0].arguments == R"({"path":"test.txt"})");
}

// ===================================================================
// build_chat_request tests
// ===================================================================

TEST_CASE("build_chat_request basic envelope", "[client]") {
    Client client("http://example.com/v1");
    json messages = {{{"role", "user"}, {"content", "hi"}}};
    json tools = json::array();

    json payload = client.build_chat_request("test-model", messages, tools, true);

    CHECK(payload["model"] == "test-model");
    CHECK(payload["messages"] == messages);
    CHECK(payload["tools"] == tools);
    CHECK(payload["stream"] == true);
    CHECK(payload.contains("stream_options"));
    CHECK(payload["stream_options"]["include_usage"] == true);
    // Without thinking, should have max_tokens
    CHECK(payload.contains("max_tokens"));
    CHECK_FALSE(payload.contains("reasoning_effort"));
    CHECK_FALSE(payload.contains("thinking"));
}

TEST_CASE("build_chat_request with thinking enabled for non-OpenAI", "[client]") {
    Client client("http://example.com/v1");
    json messages = {{{"role", "user"}, {"content", "think"}}};

    json payload = client.build_chat_request("deepseek-v4", messages, json::array(),
                                             true, 4096, 0, true);

    CHECK(payload["model"] == "deepseek-v4");
    // Non-OpenAI models use "thinking" param
    CHECK(payload.contains("thinking"));
    CHECK(payload["thinking"]["type"] == "enabled");
    // Should use max_completion_tokens instead of max_tokens
    CHECK(payload.contains("max_completion_tokens"));
    CHECK(payload["max_completion_tokens"] == 4096);
}

TEST_CASE("build_chat_request with thinking for OpenAI model", "[client]") {
    Client client("http://example.com/v1");
    json messages = {{{"role", "user"}, {"content", "think"}}};

    json payload = client.build_chat_request("o3-mini", messages, json::array(),
                                             true, 8192, 0, true);

    // OpenAI models use "reasoning_effort" param
    CHECK(payload.contains("reasoning_effort"));
    CHECK(payload["reasoning_effort"] == "high");
    CHECK(payload.contains("max_completion_tokens"));
    CHECK(payload["max_completion_tokens"] == 8192);
}

TEST_CASE("build_chat_request derives max_tokens from context_limit", "[client]") {
    Client client("http://example.com/v1");
    json messages = {{{"role", "user"}, {"content", "hi"}}};

    // When max_tokens_hint is 0 and context_limit is 32000, should use 32000/4 = 8000
    json payload = client.build_chat_request("test", messages, json::array(),
                                             true, 0, 32000, false);
    CHECK(payload["max_tokens"] == 8000);
}

// ===================================================================
// model_supports_thinking tests
// ===================================================================

TEST_CASE("model_supports_thinking known models", "[client]") {
    CHECK(Client::model_supports_thinking("deepseek-v4"));
    CHECK(Client::model_supports_thinking("o1-preview"));
    CHECK(Client::model_supports_thinking("o3-mini"));
    CHECK(Client::model_supports_thinking("o4-mini"));
    CHECK(Client::model_supports_thinking("gpt-5-turbo"));
    CHECK(Client::model_supports_thinking("qwen3-8b"));
    CHECK(Client::model_supports_thinking("glm-5"));
    CHECK(Client::model_supports_thinking("mimo-v2.5"));
}

TEST_CASE("model_supports_thinking unknown models", "[client]") {
    CHECK_FALSE(Client::model_supports_thinking("gpt-4o"));
    CHECK_FALSE(Client::model_supports_thinking("claude-3"));
    CHECK_FALSE(Client::model_supports_thinking("llama-3"));
    CHECK_FALSE(Client::model_supports_thinking("gemini-pro"));
    CHECK_FALSE(Client::model_supports_thinking(""));
}
