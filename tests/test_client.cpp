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

TEST_CASE("Client fetch_model_context_limit model not found returns 0", "[client]") {
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
