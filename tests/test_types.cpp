#include <llmclient/types.h>

#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>

using json = nlohmann::json;
using namespace llmclient;

// ========================================================================
// ToolAccumulator tests
// ========================================================================

TEST_CASE("ToolAccumulator single chunk with all fields", "[types][toolacc]") {
    ToolAccumulator acc;
    json delta = json::parse(R"({
        "tool_calls": [{
            "index": 0,
            "id": "call_abc",
            "type": "function",
            "function": {
                "name": "read_file",
                "arguments": "{\"path\": \"test.txt\"}"
            }
        }]
    })");
    acc.apply(delta);

    REQUIRE(acc.has_calls());
    auto calls = acc.finalize();
    REQUIRE(calls.size() == 1);
    CHECK(calls[0].index == 0);
    CHECK(calls[0].id == "call_abc");
    CHECK(calls[0].name == "read_file");
    CHECK(calls[0].arguments == R"({"path": "test.txt"})");
}

TEST_CASE("ToolAccumulator multi-chunk arguments concatenation",
          "[types][toolacc]") {
    ToolAccumulator acc;

    // Chunk 1: id + name + empty args
    acc.apply(json::parse(R"({
        "tool_calls": [{
            "index": 0,
            "id": "call_xyz",
            "function": {
                "name": "read_file",
                "arguments": ""
            }
        }]
    })"));

    // Chunk 2: partial args fragment
    acc.apply(json::parse(R"({
        "tool_calls": [{
            "index": 0,
            "function": {
                "arguments": "{\"path\":"
            }
        }]
    })"));

    // Chunk 3: rest of args
    acc.apply(json::parse(R"({
        "tool_calls": [{
            "index": 0,
            "function": {
                "arguments": " \"/etc/hosts\"}"
            }
        }]
    })"));

    REQUIRE(acc.has_calls());
    auto calls = acc.finalize();
    REQUIRE(calls.size() == 1);
    CHECK(calls[0].id == "call_xyz");
    CHECK(calls[0].name == "read_file");
    CHECK(calls[0].arguments == R"({"path": "/etc/hosts"})");
}

TEST_CASE("ToolAccumulator no tool_calls in delta", "[types][toolacc]") {
    ToolAccumulator acc;
    json delta = json::parse(R"({"content": "hello"})");
    acc.apply(delta);
    CHECK_FALSE(acc.has_calls());
}

TEST_CASE("ToolAccumulator empty delta", "[types][toolacc]") {
    ToolAccumulator acc;
    json delta = json::object();
    acc.apply(delta);
    CHECK_FALSE(acc.has_calls());
}

TEST_CASE("ToolAccumulator multiple parallel tool calls", "[types][toolacc]") {
    ToolAccumulator acc;

    // Chunk with two tool call starts
    acc.apply(json::parse(R"({
        "tool_calls": [
            {"index": 0, "id": "call_1", "function": {"name": "read_file", "arguments": "{\"pat"}},
            {"index": 1, "id": "call_2", "function": {"name": "grep_files", "arguments": "{\"patt"}}
        ]
    })"));

    // Chunk with continued args
    acc.apply(json::parse(R"({
        "tool_calls": [
            {"index": 0, "function": {"arguments": "h\": \"test.txt\"}"}},
            {"index": 1, "function": {"arguments": "ern\": \"foo\"}"}}
        ]
    })"));

    auto calls = acc.finalize();
    REQUIRE(calls.size() == 2);

    // finalize() returns calls sorted by index
    CHECK(calls[0].index == 0);
    CHECK(calls[0].id == "call_1");
    CHECK(calls[0].name == "read_file");
    CHECK(calls[0].arguments == R"({"path": "test.txt"})");

    CHECK(calls[1].index == 1);
    CHECK(calls[1].id == "call_2");
    CHECK(calls[1].name == "grep_files");
    CHECK(calls[1].arguments == R"({"pattern": "foo"})");
}

// ========================================================================
// ContentPart tests
// ========================================================================

TEST_CASE("ContentPart defaults to text type with empty fields", "[types][contentpart]") {
    ContentPart cp;
    CHECK(cp.type == ContentPartType::Text);
    CHECK(cp.text.empty());
    CHECK(cp.data.empty());
    CHECK(cp.media_type.empty());
    CHECK(cp.detail.empty());
}

TEST_CASE("ContentPart text JSON round-trip", "[types][contentpart]") {
    ContentPart cp;
    cp.type = ContentPartType::Text;
    cp.text = "Hello world";

    json j = cp;
    ContentPart cp2 = j.get<ContentPart>();
    CHECK(cp2.type == ContentPartType::Text);
    CHECK(cp2.text == "Hello world");
}

TEST_CASE("ContentPart image JSON round-trip", "[types][contentpart]") {
    ContentPart cp;
    cp.type = ContentPartType::Image;
    cp.data = "base64data";
    cp.media_type = "image/png";
    cp.detail = "high";

    json j = cp;
    ContentPart cp2 = j.get<ContentPart>();
    CHECK(cp2.type == ContentPartType::Image);
    CHECK(cp2.data == "base64data");
    CHECK(cp2.media_type == "image/png");
    CHECK(cp2.detail == "high");
}

TEST_CASE("ContentPart image without detail", "[types][contentpart]") {
    ContentPart cp;
    cp.type = ContentPartType::Image;
    cp.data = "data";
    cp.media_type = "image/jpeg";

    json j = cp;
    ContentPart cp2 = j.get<ContentPart>();
    CHECK(cp2.type == ContentPartType::Image);
    CHECK(cp2.detail.empty());
}

TEST_CASE("build_content_array with mixed text and image", "[types][contentpart]") {
    ContentPart text_part;
    text_part.type = ContentPartType::Text;
    text_part.text = "Look at this:";

    ContentPart img_part;
    img_part.type = ContentPartType::Image;
    img_part.data = "abc123";
    img_part.media_type = "image/png";

    auto arr = build_content_array({text_part, img_part});
    REQUIRE(arr.is_array());
    REQUIRE(arr.size() == 2);

    CHECK(arr[0]["type"] == "text");
    CHECK(arr[0]["text"] == "Look at this:");

    CHECK(arr[1]["type"] == "image_url");
    CHECK(arr[1]["image_url"]["url"] == "data:image/png;base64,abc123");
    CHECK(arr[1]["image_url"].find("detail") == arr[1]["image_url"].end());
}

TEST_CASE("build_content_array with only text", "[types][contentpart]") {
    ContentPart text_part;
    text_part.type = ContentPartType::Text;
    text_part.text = "hello";

    auto arr = build_content_array({text_part});
    REQUIRE(arr.size() == 1);
    CHECK(arr[0]["text"] == "hello");
}

TEST_CASE("build_content_array with empty parts", "[types][contentpart]") {
    auto arr = build_content_array({});
    REQUIRE(arr.is_array());
    CHECK(arr.empty());
}

TEST_CASE("has_multipart_content with single text returns false", "[types][contentpart]") {
    CHECK_FALSE(has_multipart_content({ContentPart{}}));
}

TEST_CASE("has_multipart_content with single image returns true", "[types][contentpart]") {
    ContentPart img;
    img.type = ContentPartType::Image;
    CHECK(has_multipart_content({img}));
}

TEST_CASE("has_multipart_content with multiple parts returns true", "[types][contentpart]") {
    ContentPart a, b;
    CHECK(has_multipart_content({a, b}));
}

TEST_CASE("has_multipart_content with empty vector returns false", "[types][contentpart]") {
    CHECK_FALSE(has_multipart_content({}));
}

TEST_CASE("any_user_multipart detects multipart user messages", "[types][contentpart]") {
    std::vector<std::string> roles = {"user", "assistant", "user"};
    ContentPart img;
    img.type = ContentPartType::Image;
    std::vector<std::vector<ContentPart>> parts = {
        {ContentPart{}},          // user 0: text only
        {},                       // assistant: no parts
        {img}                     // user 1: has image
    };
    CHECK(any_user_multipart(roles, parts));
}

TEST_CASE("any_user_multipart with no multipart returns false", "[types][contentpart]") {
    std::vector<std::string> roles = {"user", "assistant", "user"};
    std::vector<std::vector<ContentPart>> parts = {
        {ContentPart{}},
        {},
        {ContentPart{}}
    };
    CHECK_FALSE(any_user_multipart(roles, parts));
}

// ========================================================================
// Usage tests
// ========================================================================

TEST_CASE("Usage default values", "[types][usage]") {
    Usage u;
    CHECK(u.prompt_tokens == 0);
    CHECK(u.completion_tokens == 0);
    CHECK(u.total_tokens == 0);
}

TEST_CASE("Usage from_json with sample OpenAI response", "[types][usage]") {
    json j = json::parse(R"({
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "total_tokens": 30
    })");
    Usage u = j.get<Usage>();
    CHECK(u.prompt_tokens == 10);
    CHECK(u.completion_tokens == 20);
    CHECK(u.total_tokens == 30);
}

TEST_CASE("Usage from_json with missing fields defaults to zero", "[types][usage]") {
    json j = json::parse(R"({"total_tokens": 5})");
    Usage u = j.get<Usage>();
    CHECK(u.prompt_tokens == 0);
    CHECK(u.completion_tokens == 0);
    CHECK(u.total_tokens == 5);
}

TEST_CASE("Usage from_json with empty object", "[types][usage]") {
    json j = json::object();
    Usage u = j.get<Usage>();
    CHECK(u.prompt_tokens == 0);
    CHECK(u.completion_tokens == 0);
    CHECK(u.total_tokens == 0);
}

// ========================================================================
// ToolCall tests
// ========================================================================

TEST_CASE("ToolCall default construction", "[types][toolcall]") {
    ToolCall tc;
    CHECK(tc.index == 0);
    CHECK(tc.id.empty());
    CHECK(tc.name.empty());
    CHECK(tc.arguments.empty());
}

// ========================================================================
// sanitize_utf8 tests
// ========================================================================

TEST_CASE("sanitize_utf8 valid ASCII passthrough", "[types][utf8]") {
    CHECK(sanitize_utf8("hello world") == "hello world");
}

TEST_CASE("sanitize_utf8 valid multi-byte sequences preserved", "[types][utf8]") {
    // 2-byte: U+00E9 (é)
    CHECK(sanitize_utf8("\xC3\xA9") == "\xC3\xA9");
    // 3-byte: U+4E2D (中)
    CHECK(sanitize_utf8("\xE4\xB8\xAD") == "\xE4\xB8\xAD");
    // 4-byte: U+1F600 (😀)
    CHECK(sanitize_utf8("\xF0\x9F\x98\x80") == "\xF0\x9F\x98\x80");
}

TEST_CASE("sanitize_utf8 invalid continuation byte replaced", "[types][utf8]") {
    // 2-byte start byte followed by invalid continuation (0x00)
    std::string result = sanitize_utf8("\xC3\x00");
    CHECK(result == "\xEF\xBF\xBD\x00");
}

TEST_CASE("sanitize_utf8 truncated sequence replaced", "[types][utf8]") {
    // 3-byte start with only 1 continuation byte.
    // The start byte is invalid (no 2nd continuation), and the orphaned
    // continuation byte is also invalid on its own, so both get replaced.
    std::string result = sanitize_utf8(std::string("\xE4\xB8", 2));
    // Should produce two replacement characters (U+FFFD each = 6 bytes total)
    CHECK(result.size() == 6);
    CHECK(static_cast<unsigned char>(result[0]) == 0xEF);
    CHECK(static_cast<unsigned char>(result[1]) == 0xBF);
    CHECK(static_cast<unsigned char>(result[2]) == 0xBD);
    CHECK(static_cast<unsigned char>(result[3]) == 0xEF);
    CHECK(static_cast<unsigned char>(result[4]) == 0xBF);
    CHECK(static_cast<unsigned char>(result[5]) == 0xBD);
}

TEST_CASE("sanitize_utf8 invalid start byte replaced", "[types][utf8]") {
    // 0xFE is not a valid start byte
    std::string result = sanitize_utf8("\xFE");
    CHECK(result == "\xEF\xBF\xBD");
}

TEST_CASE("sanitize_utf8 empty string", "[types][utf8]") {
    CHECK(sanitize_utf8("").empty());
}

TEST_CASE("sanitize_utf8 mixed valid and invalid sequences", "[types][utf8]") {
    // a + é + invalid start + 😀 + truncated 2-byte start
    // Expected: a(1) + é(2) + replacement(3) + 😀(4) + replacement(3) = 13 bytes
    std::string input = "a\xC3\xA9\xFE\xF0\x9F\x98\x80\xC3";
    std::string result = sanitize_utf8(input);
    REQUIRE(result.size() == 13);
    // a at position 0
    CHECK(result[0] == 'a');
    // é at positions 1-2
    // First replacement U+FFFD at positions 3-5
    CHECK(static_cast<unsigned char>(result[3]) == 0xEF);
    CHECK(static_cast<unsigned char>(result[4]) == 0xBF);
    CHECK(static_cast<unsigned char>(result[5]) == 0xBD);
    // 😀 at positions 6-9
    // Second replacement U+FFFD at positions 10-12
    CHECK(static_cast<unsigned char>(result[10]) == 0xEF);
    CHECK(static_cast<unsigned char>(result[11]) == 0xBF);
    CHECK(static_cast<unsigned char>(result[12]) == 0xBD);
}

// ========================================================================
// SSEParser tests
// ========================================================================

TEST_CASE("SSEParser single complete event", "[types][sse]") {
    std::vector<json> received;
    bool done = false;

    SSEParser parser(SSEParser::Callbacks{
        .on_data = [&](const std::string&, const json& j) { received.push_back(j); },
        .on_done = [&]() { done = true; },
        .on_error = [&](const std::string& e) { FAIL(e); },
    });

    parser.feed("data: {\"key\":\"value\"}\n\n", 24);
    REQUIRE(received.size() == 1);
    CHECK(received[0]["key"] == "value");
    CHECK_FALSE(done);
}

TEST_CASE("SSEParser multiple events in one feed", "[types][sse]") {
    std::vector<json> received;
    bool done = false;

    SSEParser parser(SSEParser::Callbacks{
        .on_data = [&](const std::string&, const json& j) { received.push_back(j); },
        .on_done = [&]() { done = true; },
        .on_error = [&](const std::string& e) { FAIL(e); },
    });

    parser.feed("data: {\"a\":1}\n\ndata: {\"b\":2}\n\ndata: [DONE]\n\n", 46);
    REQUIRE(received.size() == 2);
    CHECK(received[0]["a"] == 1);
    CHECK(received[1]["b"] == 2);
    CHECK(done);
}

TEST_CASE("SSEParser partial data across feed calls", "[types][sse]") {
    std::vector<json> received;
    bool done = false;
    std::string error;

    SSEParser parser(SSEParser::Callbacks{
        .on_data = [&](const std::string&, const json& j) { received.push_back(j); },
        .on_done = [&]() { done = true; },
        .on_error = [&](const std::string& e) { error = e; },
    });

    parser.feed("data: {\"k", 9);
    CHECK(received.empty());

    parser.feed("ey\":1}\n\n", 8);
    REQUIRE(received.size() == 1);
    CHECK(received[0]["key"] == 1);
}

TEST_CASE("SSEParser ignores non-data lines", "[types][sse]") {
    std::vector<json> received;

    SSEParser parser(SSEParser::Callbacks{
        .on_data = [&](const std::string&, const json& j) { received.push_back(j); },
        .on_done = []() {},
        .on_error = [&](const std::string& e) { FAIL(e); },
    });

    parser.feed("event: test\ndata: {\"x\":1}\n\n", 28);
    REQUIRE(received.size() == 1);
    CHECK(received[0]["x"] == 1);
}

TEST_CASE("SSEParser reset clears state", "[types][sse]") {
    std::vector<json> received;

    SSEParser parser(SSEParser::Callbacks{
        .on_data = [&](const std::string&, const json& j) { received.push_back(j); },
        .on_done = []() {},
        .on_error = [&](const std::string& e) { FAIL(e); },
    });

    parser.feed("data: {\"a\":1}", 13);  // incomplete (no \n), stays buffered
    parser.reset();
    parser.feed("data: {\"b\":2}\n\n", 15);

    REQUIRE(received.size() == 1);
    CHECK(received[0]["b"] == 2);
}

TEST_CASE("SSEParser malformed JSON calls on_error", "[types][sse]") {
    bool errored = false;

    SSEParser parser(SSEParser::Callbacks{
        .on_data = [](const std::string&, const json&) {},
        .on_done = []() {},
        .on_error = [&](const std::string&) { errored = true; },
    });

    parser.feed("data: {invalid}\n\n", 17);
    CHECK(errored);
}

TEST_CASE("SSEParser flush processes remaining buffered data", "[types][sse]") {
    std::vector<json> received;
    bool done = false;

    SSEParser parser(SSEParser::Callbacks{
        .on_data = [&](const std::string&, const json& j) { received.push_back(j); },
        .on_done = [&]() { done = true; },
        .on_error = [&](const std::string& e) { FAIL(e); },
    });

    // Feed partial data: first event is complete (has \n\n), [DONE] lacks trailing \n
    parser.feed("data: {\"key\":\"value\"}\n\ndata: [DONE]", 35);

    // The first event was already processed by line-splitting;
    // "data: [DONE]" has no trailing \n so it stays buffered.
    REQUIRE(received.size() == 1);
    CHECK(received[0]["key"] == "value");
    CHECK_FALSE(done);

    // flush() processes the remaining "[DONE]" line
    parser.flush();

    CHECK(received.size() == 1);  // no new data events
    CHECK(done);                  // [DONE] was processed
}

TEST_CASE("SSEParser flush handles partial non-DONE data", "[types][sse]") {
    std::vector<json> received;
    bool done = false;
    std::string error;

    SSEParser parser(SSEParser::Callbacks{
        .on_data = [&](const std::string&, const json& j) { received.push_back(j); },
        .on_done = [&]() { done = true; },
        .on_error = [&](const std::string& e) { error = e; },
    });

    // A content line without trailing \n (connection closed mid-stream)
    parser.feed("data: {\"msg\":\"hello\"}", 21);
    CHECK(received.empty());  // nothing processed yet

    parser.flush();
    REQUIRE(received.size() == 1);
    CHECK(received[0]["msg"] == "hello");
    CHECK_FALSE(done);
    CHECK(error.empty());
}

TEST_CASE("SSEParser flush with no buffered data is safe", "[types][sse]") {
    SSEParser parser(SSEParser::Callbacks{
        .on_data = [](const std::string&, const json&) {},
        .on_done = []() {},
        .on_error = [](const std::string&) {},
    });

    // flush with empty buffer should not crash
    parser.flush();
    // flush after well-formed complete data should not crash
    parser.feed("data: {\"x\":1}\n\n", 13);
    parser.flush();
    // flush after reset should not crash
    parser.reset();
    parser.flush();
}

TEST_CASE("SSEParser raw accumulated data", "[types][sse]") {
    SSEParser parser(SSEParser::Callbacks{
        .on_data = [](const std::string&, const json&) {},
        .on_done = []() {},
        .on_error = [](const std::string&) {},
    });

    parser.feed("data: {\"a\":1}\n\n", 16);
    // raw() should contain the accumulated input
    CHECK(parser.raw().find("data:") == 0);
    CHECK(parser.raw().find("{\"a\":1}") != std::string::npos);
}

// ========================================================================
// ProtocolMessage tests
// ========================================================================

TEST_CASE("ProtocolMessage roles", "[types][protocol]") {
    ProtocolMessage sys;
    sys.role = "system";
    sys.content = "You are helpful.";
    CHECK(sys.role == "system");
    CHECK(sys.content.value_or("") == "You are helpful.");
    CHECK(sys.tool_calls.empty());

    ProtocolMessage user;
    user.role = "user";
    user.content = "Hello";
    CHECK(user.role == "user");

    ProtocolMessage asst;
    asst.role = "assistant";
    asst.content = std::nullopt;
    ToolCall tc;
    tc.id = "call_1";
    tc.name = "test";
    tc.arguments = "{}";
    asst.tool_calls.push_back(tc);
    CHECK(asst.role == "assistant");
    CHECK_FALSE(asst.content.has_value());
    CHECK(asst.tool_calls.size() == 1);

    ProtocolMessage tool;
    tool.role = "tool";
    tool.tool_call_id = "call_1";
    tool.content = "result";
    CHECK(tool.role == "tool");
    CHECK(tool.content.value_or("") == "result");
}

TEST_CASE("ProtocolMessage with tool_calls has null content", "[types][protocol]") {
    ProtocolMessage pm;
    pm.role = "assistant";
    pm.content = std::nullopt;
    pm.tool_calls.resize(1);
    CHECK_FALSE(pm.content.has_value());
}

// ========================================================================
// make_function_tool tests
// ========================================================================

TEST_CASE("make_function_tool produces correct JSON shape", "[types][tool]") {
    json params = {{"type", "object"}, {"properties", {{"path", {{"type", "string"}}}}}};
    json tool = make_function_tool("read_file", "Read a file", params);

    CHECK(tool["type"] == "function");
    CHECK(tool["function"]["name"] == "read_file");
    CHECK(tool["function"]["description"] == "Read a file");
    CHECK(tool["function"]["parameters"]["type"] == "object");
    CHECK(tool["function"]["parameters"]["properties"]["path"]["type"] == "string");
}

TEST_CASE("make_function_tool with empty description", "[types][tool]") {
    json tool = make_function_tool("noop", "", json::object());
    CHECK(tool["type"] == "function");
    CHECK(tool["function"]["name"] == "noop");
    CHECK(tool["function"]["description"] == "");
}
