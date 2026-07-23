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

// Helper to build a simple ChatRequest for testing.
static ChatRequest simple_req(const std::string& model = "test",
                              const std::string& user_msg = "hi",
                              bool stream = false) {
    ChatRequest req;
    req.model = model;
    req.stream = stream;
    ProtocolMessage pm;
    pm.role = "user";
    pm.content = user_msg;
    req.messages.push_back(std::move(pm));
    return req;
}

// ===================================================================
// Non-streaming chat
// ===================================================================

TEST_CASE("Client non-streaming returns ChatResponse", "[client]") {
    MockServer server([](const std::string&) -> std::string {
        return R"({"id":"test-1","choices":[{"message":{"role":"assistant","content":"Hello!"}}]})";
    });

    Client client(server.base_url());
    auto req = simple_req();

    auto result = client.chat(req);
    REQUIRE(result);
    CHECK(result->id == "test-1");
    CHECK(result->content == "Hello!");
}

TEST_CASE("Client non-streaming errors on bad URL", "[client]") {
    Client client("http://127.0.0.1:1/v1");
    auto req = simple_req();

    auto result = client.chat(req);
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
    auto req = simple_req();
    auto result = client.chat(req);
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
    auto req = simple_req();
    auto result = client.chat(req);
    REQUIRE(result);
    CHECK_FALSE(auth_seen);
}

// ===================================================================
// Streaming chat
// ===================================================================

TEST_CASE("Client streaming calls on_delta", "[client]") {
    MockServer server(
        [](const std::string&) -> std::string {
            return "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\ndata: {\"choices\":[{\"delta\":{\"content\":\" world\"}}]}\n\ndata: [DONE]\n\n";
        },
        true);

    Client client(server.base_url());

    std::string content;
    bool done = false;
    bool errored = false;

    SSEParser::Callbacks cbs;
    cbs.on_delta = [&](const StreamDelta& d) {
        if (d.content) content += *d.content;
    };
    cbs.on_done = [&]() { done = true; };
    cbs.on_error = [&](const std::string&) { errored = true; };

    auto req = simple_req("test", "hi", true);
    auto result = client.stream_chat(req, std::move(cbs));
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

    bool errored = false;
    SSEParser::Callbacks cbs;
    cbs.on_delta = [](const StreamDelta&) {};
    cbs.on_done = []() {};
    cbs.on_error = [&](const std::string&) { errored = true; };

    auto req = simple_req("test", "", true);
    auto result = client.stream_chat(req, std::move(cbs));
    // The transfer should succeed (HTTP 200), but the parse error fires
    CHECK(result);
    CHECK(errored);
}

// ===================================================================
// Structured streaming callbacks through Client
// ===================================================================

TEST_CASE("Client streaming on_delta fires correctly", "[client]") {
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

    std::string content;
    std::string reasoning;
    int tool_call_count = 0;
    ToolAccumulator tool_acc;
    bool usage_seen = false;
    bool done = false;
    bool errored = false;

    SSEParser::Callbacks cbs;
    cbs.on_delta = [&](const StreamDelta& d) {
        if (d.content) content += *d.content;
        if (d.reasoning_content) reasoning += *d.reasoning_content;
        if (d.tool_calls) {
            for (const auto& tc : *d.tool_calls) {
                tool_call_count++;
                tool_acc.apply(tc);
            }
        }
    };
    cbs.on_done = [&]() { done = true; };
    cbs.on_error = [&](const std::string&) { errored = true; };
    cbs.on_usage = [&](Usage u) {
        usage_seen = true;
        CHECK(u.prompt_tokens == 10);
        CHECK(u.completion_tokens == 20);
        CHECK(u.total_tokens == 30);
    };

    auto req = simple_req("test", "", true);
    auto result = client.stream_chat(req, std::move(cbs));
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
}

// ===================================================================
// last_raw_response
// ===================================================================

TEST_CASE("Client last_raw_response after chat", "[client]") {
    MockServer server([](const std::string&) -> std::string {
        return R"({"id":"test-raw","choices":[{"message":{"role":"assistant","content":"raw"}}]})";
    });

    Client client(server.base_url());
    auto req = simple_req();
    auto result = client.chat(req);
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
// All supported context window field name patterns
// ===================================================================

TEST_CASE("fetch_model_context_limit reads context_window", "[client]") {
    MockServer server([](const std::string& req) -> std::string {
        if (req.find("/v1/models") != std::string::npos)
            return R"({"data":[{"id":"m","context_window":16384}]})";
        return "";
    });
    Client client(server.base_url());
    CHECK(client.fetch_model_context_limit("m") == 16384);
}

TEST_CASE("fetch_model_context_limit reads max_model_len", "[client]") {
    MockServer server([](const std::string& req) -> std::string {
        if (req.find("/v1/models") != std::string::npos)
            return R"({"data":[{"id":"m","max_model_len":32768}]})";
        return "";
    });
    Client client(server.base_url());
    CHECK(client.fetch_model_context_limit("m") == 32768);
}

TEST_CASE("fetch_model_context_limit reads max_context_length", "[client]") {
    MockServer server([](const std::string& req) -> std::string {
        if (req.find("/v1/models") != std::string::npos)
            return R"({"data":[{"id":"m","max_context_length":8192}]})";
        return "";
    });
    Client client(server.base_url());
    CHECK(client.fetch_model_context_limit("m") == 8192);
}

TEST_CASE("fetch_model_context_limit reads context_length", "[client]") {
    MockServer server([](const std::string& req) -> std::string {
        if (req.find("/v1/models") != std::string::npos)
            return R"({"data":[{"id":"m","context_length":4096}]})";
        return "";
    });
    Client client(server.base_url());
    CHECK(client.fetch_model_context_limit("m") == 4096);
}

TEST_CASE("fetch_model_context_limit reads inputTokenLimit", "[client]") {
    MockServer server([](const std::string& req) -> std::string {
        if (req.find("/v1/models") != std::string::npos)
            return R"({"data":[{"id":"m","inputTokenLimit":128000}]})";
        return "";
    });
    Client client(server.base_url());
    CHECK(client.fetch_model_context_limit("m") == 128000);
}

TEST_CASE("fetch_model_context_limit reads max_input_tokens", "[client]") {
    MockServer server([](const std::string& req) -> std::string {
        if (req.find("/v1/models") != std::string::npos)
            return R"({"data":[{"id":"m","max_input_tokens":100000}]})";
        return "";
    });
    Client client(server.base_url());
    CHECK(client.fetch_model_context_limit("m") == 100000);
}

TEST_CASE("fetch_model_context_limit reads max_total_tokens", "[client]") {
    MockServer server([](const std::string& req) -> std::string {
        if (req.find("/v1/models") != std::string::npos)
            return R"({"data":[{"id":"m","max_total_tokens":64000}]})";
        return "";
    });
    Client client(server.base_url());
    CHECK(client.fetch_model_context_limit("m") == 64000);
}

TEST_CASE("fetch_model_context_limit zero on unrecognized field", "[client]") {
    MockServer server([](const std::string& req) -> std::string {
        if (req.find("/v1/models") != std::string::npos)
            return R"({"data":[{"id":"m","some_unknown_field":999}]})";
        return "";
    });
    Client client(server.base_url());
    CHECK(client.fetch_model_context_limit("m") == 0);
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
    CHECK(client.fetch_model_context_limit("test-model") == 16000);
    CHECK(call_count == 1);

    // Second call — should be cached, no server hit
    CHECK(client.fetch_model_context_limit("test-model") == 16000);
    CHECK(call_count == 1);
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
    auto req = simple_req();
    auto result = client.chat(req);

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
    auto req = simple_req();
    auto result = client.chat(req);

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
    auto req = simple_req();
    auto result = client.chat(req);

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

    std::string content;
    SSEParser::Callbacks cbs;
    cbs.on_delta = [&](const StreamDelta& d) {
        if (d.content) {
            content += *d.content;
            // Cancel after receiving first data
            *cancelled = true;
        }
    };
    cbs.on_done = []() {};
    cbs.on_error = [](const std::string&) {};

    auto req = simple_req("test", "", true);
    // The stream might return an error due to cancellation, or succeed
    // but with cancelled token. Either is acceptable.
    auto result = client.stream_chat(req, std::move(cbs));
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

    ToolAccumulator acc;
    bool done = false;

    SSEParser::Callbacks cbs;
    cbs.on_delta = [&](const StreamDelta& d) {
        if (d.tool_calls) {
            for (const auto& tc : *d.tool_calls) {
                acc.apply(tc);
            }
        }
    };
    cbs.on_done = [&]() { done = true; };
    cbs.on_error = [](const std::string&) {};

    auto req = simple_req("test", "", true);
    auto result = client.stream_chat(req, std::move(cbs));
    REQUIRE(result);

    CHECK(done);
    auto calls = acc.finalize();
    REQUIRE(calls.size() == 1);
    CHECK(calls[0].id == "call_1");
    CHECK(calls[0].name == "read_file");
    CHECK(calls[0].arguments == R"({"path":"test.txt"})");
}

// ===================================================================
// to_json(ChatRequest) tests
// ===================================================================

TEST_CASE("to_json(ChatRequest) basic envelope", "[client]") {
    auto req = simple_req("test-model", "hi", true);
    json payload = to_json(req);

    CHECK(payload["model"] == "test-model");
    REQUIRE(payload["messages"].is_array());
    CHECK(payload["messages"].size() == 1);
    CHECK(payload["messages"][0]["role"] == "user");
    CHECK(payload["messages"][0]["content"] == "hi");
    // Empty tools vector → key omitted
    CHECK_FALSE(payload.contains("tools"));
    CHECK(payload["stream"] == true);
    CHECK(payload.contains("stream_options"));
    CHECK(payload["stream_options"]["include_usage"] == true);
    // Without thinking, should have max_tokens
    CHECK(payload.contains("max_tokens"));
    CHECK_FALSE(payload.contains("reasoning_effort"));
    CHECK_FALSE(payload.contains("thinking"));
}

TEST_CASE("to_json(ChatRequest) omits empty tools vector", "[client]") {
    auto req = simple_req("test", "hi", true);
    json payload = to_json(req);
    CHECK_FALSE(payload.contains("tools"));
}

TEST_CASE("to_json(ChatRequest) includes and preserves valid tools array", "[client]") {
    ChatRequest req;
    req.model = "test";
    req.stream = false;
    ProtocolMessage pm;
    pm.role = "user";
    pm.content = "hi";
    req.messages.push_back(std::move(pm));

    ToolDef td;
    td.name = "get_weather";
    td.description = "Get weather";
    // No params — empty typed params produces the empty object schema
    req.tools.push_back(std::move(td));

    json payload = to_json(req);

    CHECK(payload.contains("tools"));
    CHECK(payload["tools"].is_array());
    CHECK(payload["tools"].size() == 1);
    CHECK(payload["tools"][0]["type"] == "function");
    CHECK(payload["tools"][0]["function"]["name"] == "get_weather");
    // Empty params must produce a proper JSON Schema object, not null
    CHECK(payload["tools"][0]["function"]["parameters"]["type"] == "object");
    CHECK(payload["tools"][0]["function"]["parameters"]["properties"].is_object());
    CHECK(payload["tools"][0]["function"]["parameters"]["required"].is_array());
}

TEST_CASE("to_json(ChatRequest) with thinking enabled for non-OpenAI", "[client]") {
    ChatRequest req;
    req.model = "deepseek-v4";
    req.stream = true;
    req.thinking_enabled = true;
    req.max_tokens = 4096;
    ProtocolMessage pm;
    pm.role = "user";
    pm.content = "think";
    req.messages.push_back(std::move(pm));

    json payload = to_json(req);

    CHECK(payload["model"] == "deepseek-v4");
    // Non-OpenAI models use "thinking" param
    CHECK(payload.contains("thinking"));
    CHECK(payload["thinking"]["type"] == "enabled");
    // Should use max_completion_tokens instead of max_tokens
    CHECK(payload.contains("max_completion_tokens"));
    CHECK(payload["max_completion_tokens"] == 4096);
}

TEST_CASE("to_json(ChatRequest) with thinking for OpenAI model", "[client]") {
    ChatRequest req;
    req.model = "o3-mini";
    req.stream = true;
    req.thinking_enabled = true;
    req.max_tokens = 8192;
    ProtocolMessage pm;
    pm.role = "user";
    pm.content = "think";
    req.messages.push_back(std::move(pm));

    json payload = to_json(req);

    // OpenAI models use "reasoning_effort" param
    CHECK(payload.contains("reasoning_effort"));
    CHECK(payload["reasoning_effort"] == "high");
    CHECK(payload.contains("max_completion_tokens"));
    CHECK(payload["max_completion_tokens"] == 8192);
}

TEST_CASE("to_json(ChatRequest) derives max_tokens from context_limit", "[client]") {
    ChatRequest req;
    req.model = "test";
    req.stream = true;
    req.context_limit = 32000;
    ProtocolMessage pm;
    pm.role = "user";
    pm.content = "hi";
    req.messages.push_back(std::move(pm));

    json payload = to_json(req);
    // When max_tokens is 0 and context_limit is 32000, should use 32000/4 = 8000
    CHECK(payload["max_tokens"] == 8000);
}

TEST_CASE("to_json(ChatRequest) with system_prompt prepends system message", "[client]") {
    ChatRequest req;
    req.model = "test";
    req.stream = false;
    req.system_prompt = "You are a helpful assistant.";
    ProtocolMessage pm;
    pm.role = "user";
    pm.content = "hi";
    req.messages.push_back(std::move(pm));

    json payload = to_json(req);

    REQUIRE(payload["messages"].is_array());
    REQUIRE(payload["messages"].size() == 2);
    CHECK(payload["messages"][0]["role"] == "system");
    CHECK(payload["messages"][0]["content"] == "You are a helpful assistant.");
    CHECK(payload["messages"][1]["role"] == "user");
    CHECK(payload["messages"][1]["content"] == "hi");
}

TEST_CASE("to_json(ChatRequest) with reasoning_effort override", "[client]") {
    ChatRequest req;
    req.model = "o3-mini";
    req.stream = true;
    req.thinking_enabled = true;
    req.reasoning_effort = "low";  // override the default "high"
    ProtocolMessage pm;
    pm.role = "user";
    pm.content = "hi";
    req.messages.push_back(std::move(pm));

    json payload = to_json(req);

    CHECK(payload["reasoning_effort"] == "low");
}

TEST_CASE("to_json(ToolDef) produces valid tool definition", "[client]") {
    // Typed path: empty params
    ToolDef td;
    td.name = "test_tool";
    td.description = "A test tool";

    json from_td;
    to_json(from_td, td);

    CHECK(from_td["type"] == "function");
    CHECK(from_td["function"]["name"] == "test_tool");
    CHECK(from_td["function"]["description"] == "A test tool");
    CHECK(from_td["function"]["parameters"]["type"] == "object");
    CHECK(from_td["function"]["parameters"]["properties"].is_object());
}

TEST_CASE("to_json(ToolDef) raw_schema path", "[client]") {
    ToolDef td;
    td.name = "raw_tool";
    td.description = "Uses raw schema";
    td.raw_schema = R"({"type":"object","properties":{"x":{"type":"integer"}},"required":["x"]})";

    json from_td;
    to_json(from_td, td);

    CHECK(from_td["function"]["parameters"]["type"] == "object");
    CHECK(from_td["function"]["parameters"]["properties"]["x"]["type"] == "integer");
    CHECK(from_td["function"]["parameters"]["required"][0] == "x");
}

TEST_CASE("Client build_chat_request delegates to to_json", "[client]") {
    Client client("http://example.com/v1");
    auto req = simple_req("test-model", "hi", true);

    json from_client = client.build_chat_request(req);
    json from_free = to_json(req);
    CHECK(from_client == from_free);
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
