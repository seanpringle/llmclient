#include <llmclient/types.h>

#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>

using json = nlohmann::json;
using namespace llmclient;

// ========================================================================
// ToolAccumulator tests
// ========================================================================

TEST_CASE("ToolAccumulator single ToolCallDelta with all fields", "[types][toolacc]") {
    ToolAccumulator acc;
    ToolCallDelta delta;
    delta.index = 0;
    delta.id = "call_abc";
    delta.name = "read_file";
    delta.arguments_fragment = R"({"path": "test.txt"})";
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
    ToolCallDelta d1;
    d1.index = 0;
    d1.id = "call_xyz";
    d1.name = "read_file";
    d1.arguments_fragment = "";
    acc.apply(d1);

    // Chunk 2: partial args fragment
    ToolCallDelta d2;
    d2.index = 0;
    d2.arguments_fragment = R"({"path":)";
    acc.apply(d2);

    // Chunk 3: rest of args
    ToolCallDelta d3;
    d3.index = 0;
    d3.arguments_fragment = R"( "/etc/hosts"})";
    acc.apply(d3);

    REQUIRE(acc.has_calls());
    auto calls = acc.finalize();
    REQUIRE(calls.size() == 1);
    CHECK(calls[0].id == "call_xyz");
    CHECK(calls[0].name == "read_file");
    CHECK(calls[0].arguments == R"({"path": "/etc/hosts"})");
}

TEST_CASE("ToolAccumulator empty ToolCallDelta does not create call", "[types][toolacc]") {
    ToolAccumulator acc;
    // No apply calls — should not have any calls
    CHECK_FALSE(acc.has_calls());
}

TEST_CASE("ToolAccumulator multiple parallel tool calls", "[types][toolacc]") {
    ToolAccumulator acc;

    // First chunk for both calls
    ToolCallDelta d1;
    d1.index = 0;
    d1.id = "call_1";
    d1.name = "read_file";
    d1.arguments_fragment = R"({"pat)";
    acc.apply(d1);

    ToolCallDelta d2;
    d2.index = 1;
    d2.id = "call_2";
    d2.name = "grep_files";
    d2.arguments_fragment = R"({"patt)";
    acc.apply(d2);

    // Continuation chunks
    ToolCallDelta d1b;
    d1b.index = 0;
    d1b.arguments_fragment = R"(h": "test.txt"})";
    acc.apply(d1b);

    ToolCallDelta d2b;
    d2b.index = 1;
    d2b.arguments_fragment = R"(ern": "foo"})";
    acc.apply(d2b);

    auto calls = acc.finalize();
    REQUIRE(calls.size() == 2);

    CHECK(calls[0].index == 0);
    CHECK(calls[0].id == "call_1");
    CHECK(calls[0].name == "read_file");
    CHECK(calls[0].arguments == R"({"path": "test.txt"})");

    CHECK(calls[1].index == 1);
    CHECK(calls[1].id == "call_2");
    CHECK(calls[1].name == "grep_files");
    CHECK(calls[1].arguments == R"({"pattern": "foo"})");
}

TEST_CASE("ToolAccumulator overwrites id and name when present", "[types][toolacc]") {
    ToolAccumulator acc;

    ToolCallDelta d1;
    d1.index = 0;
    d1.id = "call_old";
    d1.name = "old_name";
    d1.arguments_fragment = "{}";
    acc.apply(d1);

    // Fragment updates id and name
    ToolCallDelta d2;
    d2.index = 0;
    d2.id = "call_new";
    d2.name = "new_name";
    d2.arguments_fragment = R"({"key":"val"})";
    acc.apply(d2);

    auto calls = acc.finalize();
    REQUIRE(calls.size() == 1);
    CHECK(calls[0].id == "call_new");       // overwritten
    CHECK(calls[0].name == "new_name");     // overwritten
    CHECK(calls[0].arguments == R"({}{"key":"val"})"); // concatenated
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

TEST_CASE("Usage from_json with reasoning_tokens from completion_tokens_details",
          "[types][usage]") {
    json j = json::parse(R"({
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "total_tokens": 30,
        "completion_tokens_details": {
            "reasoning_tokens": 42
        }
    })");
    Usage u = j.get<Usage>();
    CHECK(u.prompt_tokens == 10);
    CHECK(u.completion_tokens == 20);
    CHECK(u.total_tokens == 30);
    CHECK(u.reasoning_tokens == 42);
}

TEST_CASE("Usage from_json with cache fields (Anthropic)", "[types][usage]") {
    json j = json::parse(R"({
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "total_tokens": 30,
        "cache_read_input_tokens": 100,
        "cache_creation_input_tokens": 50
    })");
    Usage u = j.get<Usage>();
    CHECK(u.cache_read_input_tokens == 100);
    CHECK(u.cache_creation_input_tokens == 50);
}

TEST_CASE("Usage from_json missing optional fields default to 0", "[types][usage]") {
    json j = json::parse(R"({"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3})");
    Usage u = j.get<Usage>();
    CHECK(u.prompt_tokens == 1);
    CHECK(u.completion_tokens == 2);
    CHECK(u.total_tokens == 3);
    CHECK(u.reasoning_tokens == 0);
    CHECK(u.cache_read_input_tokens == 0);
    CHECK(u.cache_creation_input_tokens == 0);
}

// ========================================================================
// ChatResponse tests
// ========================================================================

TEST_CASE("ChatResponse default values", "[types][chatresponse]") {
    ChatResponse r;
    CHECK(r.content.empty());
    CHECK(r.reasoning_content.empty());
    CHECK(r.tool_calls.empty());
    CHECK(r.finish_reason.empty());
    CHECK(r.id.empty());
    CHECK(r.model.empty());
    CHECK(r.usage.prompt_tokens == 0);
}

TEST_CASE("ChatResponse from_json basic content", "[types][chatresponse]") {
    json j = json::parse(R"({
        "id": "chatcmpl-123",
        "model": "gpt-4",
        "choices": [{
            "index": 0,
            "finish_reason": "stop",
            "message": {
                "role": "assistant",
                "content": "Hello world"
            }
        }],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30
        }
    })");
    ChatResponse r = j.get<ChatResponse>();
    CHECK(r.id == "chatcmpl-123");
    CHECK(r.model == "gpt-4");
    CHECK(r.content == "Hello world");
    CHECK(r.finish_reason == "stop");
    CHECK(r.usage.prompt_tokens == 10);
    CHECK(r.usage.completion_tokens == 20);
    CHECK(r.usage.total_tokens == 30);
}

TEST_CASE("ChatResponse from_json with reasoning_content", "[types][chatresponse]") {
    json j = json::parse(R"({
        "id": "chatcmpl-456",
        "model": "o3-mini",
        "choices": [{
            "index": 0,
            "finish_reason": "stop",
            "message": {
                "role": "assistant",
                "content": "Final answer",
                "reasoning_content": "Step-by-step reasoning..."
            }
        }],
        "usage": {}
    })");
    ChatResponse r = j.get<ChatResponse>();
    CHECK(r.content == "Final answer");
    CHECK(r.reasoning_content == "Step-by-step reasoning...");
}

TEST_CASE("ChatResponse from_json with tool_calls synthesizes index",
          "[types][chatresponse]") {
    json j = json::parse(R"({
        "id": "call-789",
        "model": "gpt-4",
        "choices": [{
            "index": 0,
            "finish_reason": "tool_calls",
            "message": {
                "role": "assistant",
                "content": null,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{\"city\":\"London\"}"}
                    },
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {"name": "get_time",   "arguments": "{\"tz\":\"UTC\"}"}
                    },
                    {
                        "id": "call_3",
                        "type": "function",
                        "function": {"name": "get_date",   "arguments": "{}"}
                    }
                ]
            }
        }],
        "usage": {}
    })");
    ChatResponse r = j.get<ChatResponse>();
    CHECK(r.finish_reason == "tool_calls");
    REQUIRE(r.tool_calls.size() == 3);
    // Index should be synthesized from array position
    CHECK(r.tool_calls[0].index == 0);
    CHECK(r.tool_calls[0].id == "call_1");
    CHECK(r.tool_calls[0].name == "get_weather");
    CHECK(r.tool_calls[0].arguments == R"({"city":"London"})");

    CHECK(r.tool_calls[1].index == 1);
    CHECK(r.tool_calls[1].id == "call_2");
    CHECK(r.tool_calls[1].name == "get_time");
    CHECK(r.tool_calls[1].arguments == R"({"tz":"UTC"})");

    CHECK(r.tool_calls[2].index == 2);
    CHECK(r.tool_calls[2].id == "call_3");
    CHECK(r.tool_calls[2].name == "get_date");
}

TEST_CASE("ChatResponse from_json empty choices leaves defaults", "[types][chatresponse]") {
    json j = json::parse(R"({"id":"x","model":"m","choices":[]})");
    ChatResponse r = j.get<ChatResponse>();
    CHECK(r.id == "x");
    CHECK(r.model == "m");
    CHECK(r.content.empty());
    CHECK(r.tool_calls.empty());
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
    std::vector<StreamDelta> deltas;
    bool done = false;

    SSEParser parser(SSEParser::Callbacks{
        .on_done = [&]() { done = true; },
        .on_error = [&](const std::string& e) { FAIL(e); },
        .on_delta = [&](const StreamDelta& d) { deltas.push_back(d); },
    });

    parser.feed("data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\n", 51);
    REQUIRE(deltas.size() == 1);
    CHECK(deltas[0].content.has_value());
    CHECK(*deltas[0].content == "Hello");
    CHECK_FALSE(done);
}

TEST_CASE("SSEParser multiple events in one feed", "[types][sse]") {
    std::vector<std::string> contents;
    bool done = false;

    SSEParser parser(SSEParser::Callbacks{
        .on_done = [&]() { done = true; },
        .on_error = [&](const std::string& e) { FAIL(e); },
        .on_delta = [&](const StreamDelta& d) {
            if (d.content) contents.push_back(*d.content);
        },
    });

    parser.feed("data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\ndata: {\"choices\":[{\"delta\":{\"content\":\" world\"}}]}\n\ndata: [DONE]\n\n", 117);
    REQUIRE(contents.size() == 2);
    CHECK(contents[0] == "Hello");
    CHECK(contents[1] == " world");
    CHECK(done);
}

TEST_CASE("SSEParser partial data across feed calls", "[types][sse]") {
    std::vector<std::string> contents;
    bool done = false;

    SSEParser parser(SSEParser::Callbacks{
        .on_done = [&]() { done = true; },
        .on_error = [&](const std::string& e) { FAIL(e); },
        .on_delta = [&](const StreamDelta& d) {
            if (d.content) contents.push_back(*d.content);
        },
    });

    parser.feed("data: {\"choices\":[", 18);
    CHECK(contents.empty());

    parser.feed("{\"delta\":{\"content\":\"Hello\"}}]}\n\n", 33);
    REQUIRE(contents.size() == 1);
    CHECK(contents[0] == "Hello");
}

TEST_CASE("SSEParser ignores non-data lines", "[types][sse]") {
    std::vector<std::string> contents;

    SSEParser parser(SSEParser::Callbacks{
        .on_done = []() {},
        .on_error = [&](const std::string& e) { FAIL(e); },
        .on_delta = [&](const StreamDelta& d) {
            if (d.content) contents.push_back(*d.content);
        },
    });

    parser.feed("event: test\ndata: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\n", 63);
    REQUIRE(contents.size() == 1);
    CHECK(contents[0] == "Hello");
}

TEST_CASE("SSEParser reset clears state", "[types][sse]") {
    std::vector<std::string> contents;

    SSEParser parser(SSEParser::Callbacks{
        .on_done = []() {},
        .on_error = [&](const std::string& e) { FAIL(e); },
        .on_delta = [&](const StreamDelta& d) {
            if (d.content) contents.push_back(*d.content);
        },
    });

    parser.feed("data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]", 48);  // incomplete, stays buffered
    parser.reset();
    parser.feed("data: {\"choices\":[{\"delta\":{\"content\":\"World\"}}]}\n\n", 51);

    REQUIRE(contents.size() == 1);
    CHECK(contents[0] == "World");
}

TEST_CASE("SSEParser malformed JSON calls on_error", "[types][sse]") {
    bool errored = false;

    SSEParser parser(SSEParser::Callbacks{
        .on_done = []() {},
        .on_error = [&](const std::string&) { errored = true; },
        .on_delta = [](const StreamDelta&) {},
    });

    parser.feed("data: {invalid}\n\n", 17);
    CHECK(errored);
}

TEST_CASE("SSEParser flush processes remaining buffered data", "[types][sse]") {
    std::vector<std::string> contents;
    bool done = false;

    SSEParser parser(SSEParser::Callbacks{
        .on_done = [&]() { done = true; },
        .on_error = [&](const std::string& e) { FAIL(e); },
        .on_delta = [&](const StreamDelta& d) {
            if (d.content) contents.push_back(*d.content);
        },
    });

    parser.feed("data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\ndata: [DONE]", 63);

    REQUIRE(contents.size() == 1);
    CHECK(contents[0] == "Hello");
    CHECK_FALSE(done);

    parser.flush();
    CHECK(contents.size() == 1);
    CHECK(done);
}

TEST_CASE("SSEParser flush handles partial non-DONE data", "[types][sse]") {
    std::vector<std::string> contents;
    bool errored = false;

    SSEParser parser(SSEParser::Callbacks{
        .on_done = []() {},
        .on_error = [&](const std::string&) { errored = true; },
        .on_delta = [&](const StreamDelta& d) {
            if (d.content) contents.push_back(*d.content);
        },
    });

    parser.feed("data: {\"msg\":\"hello\"}", 21);
    CHECK(contents.empty());

    parser.flush();
    CHECK(contents.empty());
    CHECK_FALSE(errored);
}

TEST_CASE("SSEParser flush with no buffered data is safe", "[types][sse]") {
    SSEParser parser(SSEParser::Callbacks{
        .on_done = []() {},
        .on_error = [](const std::string&) {},
        .on_delta = [](const StreamDelta&) {},
    });

    parser.flush();
    parser.feed("data: {\"choices\":[{\"delta\":{\"content\":\"x\"}}]}\n\n", 47);
    parser.flush();
    parser.reset();
    parser.flush();
}

TEST_CASE("SSEParser raw accumulated data", "[types][sse]") {
    SSEParser parser(SSEParser::Callbacks{
        .on_done = []() {},
        .on_error = [](const std::string&) {},
        .on_delta = [](const StreamDelta&) {},
    });

    parser.feed("data: {\"choices\":[{\"delta\":{\"content\":\"X\"}}]}\n\n", 47);
    CHECK(parser.raw().find("content") != std::string::npos);
}

TEST_CASE("SSEParser finish_reason fires on_finish", "[types][sse]") {
    std::string finish_reason;
    bool done = false;

    SSEParser parser(SSEParser::Callbacks{
        .on_done = [&]() { done = true; },
        .on_error = [&](const std::string& e) { FAIL(e); },
        .on_delta = [](const StreamDelta&) {},
        .on_finish = [&](std::string_view r) { finish_reason = std::string(r); },
    });

    parser.feed("data: {\"choices\":[{\"finish_reason\":\"stop\",\"delta\":{}}]}\n\ndata: [DONE]\n\n", 71);

    CHECK(finish_reason == "stop");
    CHECK(done);
}

TEST_CASE("SSEParser on_delta with reasoning_content", "[types][sse]") {
    std::string reasoning;
    std::string content;

    SSEParser parser(SSEParser::Callbacks{
        .on_done = []() {},
        .on_error = [&](const std::string& e) { FAIL(e); },
        .on_delta = [&](const StreamDelta& d) {
            if (d.reasoning_content) reasoning = *d.reasoning_content;
            if (d.content) content = *d.content;
        },
    });

    parser.feed("data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"Thinking...\",\"content\":\"Answer\"}}]}\n\n", 86);

    CHECK(reasoning == "Thinking...");
    CHECK(content == "Answer");
}

TEST_CASE("SSEParser on_delta with reasoning (alternate field name)", "[types][sse]") {
    std::string reasoning;

    SSEParser parser(SSEParser::Callbacks{
        .on_done = []() {},
        .on_error = [&](const std::string& e) { FAIL(e); },
        .on_delta = [&](const StreamDelta& d) {
            if (d.reasoning_content) reasoning = *d.reasoning_content;
        },
    });

    parser.feed("data: {\"choices\":[{\"delta\":{\"reasoning\":\"Thinking...\"}}]}\n\n", 59);

    CHECK(reasoning == "Thinking...");
}

TEST_CASE("SSEParser on_delta with tool_calls", "[types][sse]") {
    int delta_count = 0;
    ToolAccumulator acc;

    SSEParser parser(SSEParser::Callbacks{
        .on_done = []() {},
        .on_error = [&](const std::string& e) { FAIL(e); },
        .on_delta = [&](const StreamDelta& d) {
            if (d.tool_calls) {
                for (const auto& tc : *d.tool_calls) {
                    delta_count++;
                    acc.apply(tc);
                }
            }
        },
    });

    parser.feed("data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"function\":{\"name\":\"test\",\"arguments\":\"\"}}]}}]}\n\n"
                "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{}\"}}]}}]}\n\n", 216);

    CHECK(delta_count == 2);
    auto calls = acc.finalize();
    REQUIRE(calls.size() == 1);
    CHECK(calls[0].id == "call_1");
    CHECK(calls[0].name == "test");
    CHECK(calls[0].arguments == "{}");
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
// ChatResponse refusal parsing tests
// ========================================================================

TEST_CASE("ChatResponse from_json parses refusal", "[types][chatresponse]") {
    json j = json::parse(R"({
        "id": "chatcmpl-ref",
        "model": "gpt-4",
        "choices": [{
            "index": 0,
            "finish_reason": "stop",
            "message": {
                "role": "assistant",
                "content": null,
                "refusal": "I cannot answer that question."
            }
        }]
    })");
    ChatResponse r = j.get<ChatResponse>();
    CHECK(r.content.empty());
    CHECK(r.refusal == "I cannot answer that question.");
}

TEST_CASE("ChatResponse from_json refusal empty when absent", "[types][chatresponse]") {
    json j = json::parse(R"({
        "id": "chatcmpl-123",
        "model": "gpt-4",
        "choices": [{
            "index": 0,
            "finish_reason": "stop",
            "message": {
                "role": "assistant",
                "content": "Hello"
            }
        }]
    })");
    ChatResponse r = j.get<ChatResponse>();
    CHECK(r.content == "Hello");
    CHECK(r.refusal.empty());
}

// ========================================================================
// ChatResponse tool_call type filter tests
// ========================================================================

TEST_CASE("ChatResponse from_json skips non-function tool_calls", "[types][chatresponse]") {
    json j = json::parse(R"({
        "id": "chatcmpl-tc",
        "model": "gpt-4",
        "choices": [{
            "index": 0,
            "finish_reason": "tool_calls",
            "message": {
                "role": "assistant",
                "content": null,
                "tool_calls": [
                    {
                        "id": "call_func",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{}"}
                    },
                    {
                        "id": "call_custom",
                        "type": "custom",
                        "function": {"name": "custom_op", "arguments": "{}"}
                    }
                ]
            }
        }]
    })");
    ChatResponse r = j.get<ChatResponse>();
    REQUIRE(r.tool_calls.size() == 1);
    CHECK(r.tool_calls[0].id == "call_func");
}

// ========================================================================
// ChatResponse content array parsing
// ========================================================================

TEST_CASE("ChatResponse from_json content as array concatenates text", "[types][chatresponse]") {
    json j = json::parse(R"({
        "id": "chatcmpl-arr",
        "model": "gpt-4",
        "choices": [{
            "index": 0,
            "finish_reason": "stop",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "text", "text": "World"}
                ]
            }
        }]
    })");
    ChatResponse r = j.get<ChatResponse>();
    CHECK(r.content == "Hello\nWorld");
}

TEST_CASE("ChatResponse from_json content array skips non-text parts", "[types][chatresponse]") {
    json j = json::parse(R"({
        "id": "chatcmpl-arr2",
        "model": "gpt-4",
        "choices": [{
            "index": 0,
            "finish_reason": "stop",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Caption:"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}
                ]
            }
        }]
    })");
    ChatResponse r = j.get<ChatResponse>();
    CHECK(r.content == "Caption:");
}

// ========================================================================
// ContentPart wire format tests (B1 fix)
// ========================================================================

TEST_CASE("ContentPart to_json image writes detail inside image_url", "[types][contentpart]") {
    ContentPart cp;
    cp.type = ContentPartType::Image;
    cp.data = "base64data";
    cp.media_type = "image/png";
    cp.detail = "high";

    json j = cp;
    // Should produce wire-format: {"type":"image_url","image_url":{"url":"...","detail":"high"}}
    CHECK(j["type"] == "image_url");
    CHECK(j["image_url"]["url"] == "data:image/png;base64,base64data");
    CHECK(j["image_url"]["detail"] == "high");
    // detail should NOT be at top level
    CHECK_FALSE(j.contains("data"));
    CHECK_FALSE(j.contains("media_type"));
}

TEST_CASE("ContentPart from_json reads detail from image_url", "[types][contentpart]") {
    json j = json::parse(R"({
        "type": "image_url",
        "image_url": {
            "url": "data:image/jpeg;base64,test123",
            "detail": "low"
        }
    })");
    ContentPart cp = j.get<ContentPart>();
    CHECK(cp.type == ContentPartType::Image);
    CHECK(cp.data == "test123");
    CHECK(cp.media_type == "image/jpeg");
    CHECK(cp.detail == "low");
}

// ========================================================================
// StreamDelta finish_reason tests (G7)
// ========================================================================

TEST_CASE("SSEParser on_delta includes finish_reason", "[types][sse]") {
    std::optional<std::string> finish_reason;
    bool delta_fired = false;

    SSEParser parser(SSEParser::Callbacks{
        .on_done = []() {},
        .on_error = [](const std::string&) {},
        .on_delta = [&](const StreamDelta& d) {
            delta_fired = true;
            if (d.finish_reason) finish_reason = *d.finish_reason;
        },
    });

    parser.feed("data: {\"choices\":[{\"finish_reason\":\"stop\",\"delta\":{}}]}\n\ndata: [DONE]\n\n", 71);
    CHECK(delta_fired);
    CHECK(finish_reason.has_value());
    CHECK(*finish_reason == "stop");
}

// ========================================================================
// StreamDelta refusal in streaming (decision #6)
// ========================================================================

TEST_CASE("SSEParser on_delta includes refusal", "[types][sse]") {
    std::optional<std::string> refusal;

    SSEParser parser(SSEParser::Callbacks{
        .on_done = []() {},
        .on_error = [](const std::string&) {},
        .on_delta = [&](const StreamDelta& d) {
            if (d.refusal) refusal = *d.refusal;
        },
    });

    parser.feed("data: {\"choices\":[{\"delta\":{\"refusal\":\"Cannot answer\"}}]}\n\n", 64);
    CHECK(refusal.has_value());
    CHECK(*refusal == "Cannot answer");
}

// ========================================================================
// ProtocolMessage developer role (G8)
// ========================================================================

TEST_CASE("ProtocolMessage developer role serialized as-is", "[types][protocol]") {
    ProtocolMessage dev;
    dev.role = "developer";
    dev.content = "Think step by step.";

    ChatRequest req;
    req.model = "test";
    req.messages.push_back(std::move(dev));

    json payload = to_json(req);
    REQUIRE(payload["messages"].size() == 1);
    CHECK(payload["messages"][0]["role"] == "developer");
    CHECK(payload["messages"][0]["content"] == "Think step by step.");
}
