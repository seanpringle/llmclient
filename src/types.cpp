#include "llmclient/types.h"

#include <algorithm>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string_view>

namespace llmclient {

// ---------------------------------------------------------------------------
// Usage JSON deserialization
// ---------------------------------------------------------------------------

void from_json(const nlohmann::json& j, Usage& u) {
    u.prompt_tokens = j.value("prompt_tokens", 0);
    u.completion_tokens = j.value("completion_tokens", 0);
    u.total_tokens = j.value("total_tokens", 0);
    u.cache_read_input_tokens = j.value("cache_read_input_tokens", 0);
    u.cache_creation_input_tokens = j.value("cache_creation_input_tokens", 0);
    // reasoning_tokens is nested under completion_tokens_details
    u.reasoning_tokens = 0;
    auto details = j.find("completion_tokens_details");
    if (details != j.end() && details->is_object()) {
        u.reasoning_tokens = details->value("reasoning_tokens", 0);
    }
}

// ---------------------------------------------------------------------------
// ChatResponse JSON deserialization
// ---------------------------------------------------------------------------

void from_json(const nlohmann::json& j, ChatResponse& r) {
    r.id = j.value("id", "");
    r.model = j.value("model", "");

    // Parse usage (may be absent)
    auto usage_it = j.find("usage");
    if (usage_it != j.end() && usage_it->is_object()) {
        r.usage = usage_it->get<Usage>();
    }

    // Parse choices[0]
    auto choices = j.find("choices");
    if (choices == j.end() || !choices->is_array() || choices->empty()) {
        return; // leave defaults
    }
    const auto& choice = (*choices)[0];
    r.finish_reason = choice.value("finish_reason", "");

    // message may be absent in unconventional responses
    auto msg_it = choice.find("message");
    if (msg_it == choice.end() || !msg_it->is_object()) {
        return;
    }
    const auto& message = *msg_it;
    // content may be null (for tool_call messages)
    auto content_it = message.find("content");
    if (content_it != message.end() && !content_it->is_null()) {
        r.content = content_it->get<std::string>();
    }
    r.reasoning_content = message.value("reasoning_content", "");

    // Parse tool_calls — non-streaming has no "index" field, synthesize from position
    auto tc_it = message.find("tool_calls");
    if (tc_it != message.end() && tc_it->is_array()) {
        int idx = 0;
        for (const auto& tc_json : *tc_it) {
            ToolCall tc;
            tc.index = idx++;
            tc.id = tc_json.value("id", "");
            auto func = tc_json.find("function");
            if (func != tc_json.end() && func->is_object()) {
                tc.name = func->value("name", "");
                tc.arguments = func->value("arguments", "");
            }
            r.tool_calls.push_back(std::move(tc));
        }
    }
}

// ---------------------------------------------------------------------------
// ContentPart serialization
// ---------------------------------------------------------------------------

static const char* content_part_type_str(ContentPartType t) {
    switch (t) {
        case ContentPartType::Text: return "text";
        case ContentPartType::Image: return "image";
    }
    return "text";
}

static ContentPartType content_part_type_from_str(const std::string& s) {
    if (s == "image") return ContentPartType::Image;
    return ContentPartType::Text;
}

void to_json(nlohmann::json& j, const ContentPart& cp) {
    j["type"] = content_part_type_str(cp.type);
    if (cp.type == ContentPartType::Text) {
        j["text"] = cp.text;
    } else {
        j["data"] = cp.data;
        j["media_type"] = cp.media_type;
        if (!cp.detail.empty()) j["detail"] = cp.detail;
    }
}

void from_json(const nlohmann::json& j, ContentPart& cp) {
    cp.type = content_part_type_from_str(j.value("type", "text"));
    cp.text = j.value("text", "");
    cp.data = j.value("data", "");
    cp.media_type = j.value("media_type", "");
    cp.detail = j.value("detail", "");
}

nlohmann::json build_content_array(const std::vector<ContentPart>& parts) {
    nlohmann::json arr = nlohmann::json::array();
    for (const auto& p : parts) {
        if (p.type == ContentPartType::Text) {
            arr.push_back({{"type", "text"}, {"text", sanitize_utf8(p.text)}});
        } else if (p.type == ContentPartType::Image) {
            std::string data_url = "data:" + p.media_type + ";base64," + p.data;
            nlohmann::json img;
            img["type"] = "image_url";
            img["image_url"]["url"] = data_url;
            if (!p.detail.empty())
                img["image_url"]["detail"] = p.detail;
            arr.push_back(std::move(img));
        }
    }
    return arr;
}

bool has_multipart_content(const std::vector<ContentPart>& parts) {
    if (parts.empty())
        return false;
    if (parts.size() > 1)
        return true;
    return parts[0].type != ContentPartType::Text;
}

bool any_user_multipart(const std::vector<std::string>& roles,
                        const std::vector<std::vector<ContentPart>>& parts_list) {
    if (roles.size() != parts_list.size())
        return false;
    for (size_t i = 0; i < roles.size(); i++) {
        if (roles[i] == "user" && has_multipart_content(parts_list[i]))
            return true;
    }
    return false;
}

// ---------------------------------------------------------------------------
// ToolAccumulator — merges streaming tool_call deltas across SSE chunks
//
// OpenAI streaming format emits tool_calls as incremental delta fragments.
// Each chunk carries an "index" to identify which call it belongs to, and
// "arguments" are accumulated by appending successive string fragments.
// ---------------------------------------------------------------------------

void ToolAccumulator::apply(const nlohmann::json& delta) {
    auto it = delta.find("tool_calls");
    if (it == delta.end() || !it->is_array()) {
        return;
    }

    for (const auto& tc : *it) {
        int idx = tc.value("index", 0);
        auto& call = calls_[idx];
        call.index = idx;

        auto id_it = tc.find("id");
        if (id_it != tc.end() && id_it->is_string()) {
            call.id = id_it->get<std::string>();
        }

        auto func_it = tc.find("function");
        if (func_it != tc.end() && func_it->is_object()) {
            auto name_it = func_it->find("name");
            if (name_it != func_it->end() && name_it->is_string()) {
                call.name = name_it->get<std::string>();
            }
            auto args_it = func_it->find("arguments");
            if (args_it != func_it->end() && args_it->is_string()) {
                call.arguments += args_it->get<std::string>();
            }
        }
    }
}

std::vector<ToolCall> ToolAccumulator::finalize() const {
    std::vector<ToolCall> result;
    result.reserve(calls_.size());
    for (const auto& [idx, tc] : calls_) {
        result.push_back(tc);
    }
    std::sort(result.begin(), result.end(), [](const ToolCall& a, const ToolCall& b) { return a.index < b.index; });
    return result;
}

// ---------------------------------------------------------------------------
// is_valid_cont — helper: true if byte is a valid UTF-8 continuation byte
// ---------------------------------------------------------------------------

static bool is_valid_cont(unsigned char c) {
    return (c & 0xC0) == 0x80;
}

// ---------------------------------------------------------------------------
// sanitize_utf8 — ensure valid UTF-8 by replacing invalid sequences
// ---------------------------------------------------------------------------

std::string sanitize_utf8(const std::string& input) {
    // Validates and fixes up UTF-8 by replacing invalid byte sequences
    // with the Unicode replacement character (U+FFFD) encoded as UTF-8.
    // Based on the approach from skeleton/helpers.h.
    std::string result;
    result.reserve(input.size());

    size_t i = 0;
    while (i < input.size()) {
        unsigned char c = static_cast<unsigned char>(input[i]);

        if (c <= 0x7F) {
            // 1-byte sequence (ASCII)
            result += input[i];
            i++;
        } else if (c >= 0xC2 && c <= 0xDF) {
            // 2-byte sequence
            if (i + 1 >= input.size() || !is_valid_cont(static_cast<unsigned char>(input[i + 1]))) {
                result += "\xEF\xBF\xBD"; // replacement character
                i++;
            } else {
                result += input[i];
                result += input[i + 1];
                i += 2;
            }
        } else if (c >= 0xE0 && c <= 0xEF) {
            // 3-byte sequence
            if (i + 2 >= input.size() ||
                !is_valid_cont(static_cast<unsigned char>(input[i + 1])) ||
                !is_valid_cont(static_cast<unsigned char>(input[i + 2]))) {
                result += "\xEF\xBF\xBD";
                i++;
            } else {
                result += input[i];
                result += input[i + 1];
                result += input[i + 2];
                i += 3;
            }
        } else if (c >= 0xF0 && c <= 0xF4) {
            // 4-byte sequence
            if (i + 3 >= input.size() ||
                !is_valid_cont(static_cast<unsigned char>(input[i + 1])) ||
                !is_valid_cont(static_cast<unsigned char>(input[i + 2])) ||
                !is_valid_cont(static_cast<unsigned char>(input[i + 3]))) {
                result += "\xEF\xBF\xBD";
                i++;
            } else {
                result += input[i];
                result += input[i + 1];
                result += input[i + 2];
                result += input[i + 3];
                i += 4;
            }
        } else {
            // Invalid start byte
            result += "\xEF\xBF\xBD";
            i++;
        }
    }

    return result;
}

// ---------------------------------------------------------------------------
// SSEParser
// ---------------------------------------------------------------------------

SSEParser::SSEParser(Callbacks cb) : cb_(std::move(cb)) {}

void SSEParser::feed(const char* data, size_t len) {
    raw_.append(data, len);
    buf_.append(data, len);

    while (true) {
        auto nl = buf_.find('\n');
        if (nl == std::string::npos) {
            break; // incomplete line, wait for more data
        }

        std::string line = buf_.substr(0, nl);
        buf_.erase(0, nl + 1); // remove line + \n
        process_line(std::move(line));
    }
}

void SSEParser::process_line(std::string line) {
    // Strip trailing \r if present
    if (!line.empty() && line.back() == '\r') {
        line.pop_back();
    }

    // Skip empty lines (event separator) — reset event name
    if (line.empty()) {
        current_event_.clear();
        return;
    }

    // Track event: lines (SSE named events)
    constexpr std::string_view event_prefix = "event: ";
    if (line.size() > event_prefix.size() && std::string_view(line).substr(0, event_prefix.size()) == event_prefix) {
        current_event_ = line.substr(event_prefix.size());
        return;
    }

    // Accept "data:" with or without trailing space.
    // The SSE spec allows optional whitespace after the colon.
    // Some providers (opencode) omit the space for certain
    // events while including it for OpenAI-format chunks.
    constexpr std::string_view prefix_nospace = "data:";
    if (line.size() >= prefix_nospace.size() && std::string_view(line).substr(0, prefix_nospace.size()) == prefix_nospace) {
        // Skip "data:" and optional whitespace
        size_t payload_start = prefix_nospace.size();
        if (payload_start < line.size() && line[payload_start] == ' ')
            payload_start++;
        std::string payload = line.substr(payload_start);

        // [DONE] is OpenAI-specific; invoke on_done callback to signal stream end.
        if (payload == "[DONE]") {
            try {
                if (cb_.on_done) {
                    cb_.on_done();
                }
            } catch (...) {
                // swallow all exceptions — must not throw through C frames (libcurl)
            }
            return;
        }

        try {
            nlohmann::json j = nlohmann::json::parse(payload);
            try {
                if (cb_.on_data) {
                    cb_.on_data(current_event_, j);
                }

                // ── Structured delta extraction (OpenAI streaming format) ──
                // Only process data events that carry choices with delta.
                if (j.contains("choices") && j["choices"].is_array() && !j["choices"].empty()) {
                    const auto& delta = j["choices"][0]["delta"];

                    auto extract_str = [&](const std::string& key) -> std::string_view {
                        auto it = delta.find(key);
                        if (it != delta.end() && it->is_string()) {
                            auto& s = it->get_ref<const std::string&>();
                            return std::string_view(s);
                        }
                        return {};
                    };

                    // Content delta
                    if (cb_.on_content_delta) {
                        auto text = extract_str("content");
                        if (!text.empty())
                            cb_.on_content_delta(text);
                    }

                    // Reasoning delta — check both `reasoning_content` and `reasoning`
                    if (cb_.on_reasoning_delta) {
                        auto text = extract_str("reasoning_content");
                        if (text.empty())
                            text = extract_str("reasoning");
                        if (!text.empty())
                            cb_.on_reasoning_delta(text);
                    }

                    // Tool calls delta — pass the full delta so consumers
                    // can use ToolAccumulator::apply().
                    if (cb_.on_tool_call_delta) {
                        auto tc_it = delta.find("tool_calls");
                        if (tc_it != delta.end() && tc_it->is_array()) {
                            cb_.on_tool_call_delta(delta);
                        }
                    }
                }

                // Usage object (final chunk)
                if (cb_.on_usage) {
                    auto usage_it = j.find("usage");
                    if (usage_it != j.end() && usage_it->is_object()) {
                        try {
                            cb_.on_usage(usage_it->get<Usage>());
                        } catch (...) {
                            // swallow parse errors for usage
                        }
                    }
                }
            } catch (...) {
                // swallow — must not throw through C frames (libcurl)
            }
        } catch (const std::exception& e) {
            try {
                if (cb_.on_error) {
                    cb_.on_error(std::string("SSE error: ") + e.what() + " | payload: " + payload);
                }
            } catch (...) {
                // swallow — must not throw through C frames (libcurl)
            }
        } catch (...) {
            try {
                if (cb_.on_error) {
                    cb_.on_error("SSE error: unknown exception | payload: " + payload);
                }
            } catch (...) {
                // swallow — must not throw through C frames (libcurl)
            }
        }
        return;
    }

    // Ignore other fields (:keepalive, etc.)
}

void SSEParser::flush() {
    if (!buf_.empty()) {
        process_line(std::move(buf_));
        buf_.clear();
    }
}

void SSEParser::reset() {
    buf_.clear();
    raw_.clear();
    current_event_.clear();
}

// ---------------------------------------------------------------------------
// build_openai_payload — build OpenAI-compatible messages array
// ---------------------------------------------------------------------------

nlohmann::json build_openai_payload(const std::vector<ProtocolMessage>& messages,
                                    const std::string& system_prompt) {
    nlohmann::json arr = nlohmann::json::array();

    arr.push_back({{"role", "system"}, {"content", sanitize_utf8(system_prompt)}});

    // Check if any user message uses multipart content — forces all user
    // messages to use array form.
    bool force_multipart = false;
    for (const auto& msg : messages) {
        if (msg.role == "user" && has_multipart_content(msg.parts)) {
            force_multipart = true;
            break;
        }
    }

    for (const auto& msg : messages) {
        if (msg.role == "assistant" && !msg.tool_calls.empty()) {
            // Assistant with tool calls — expand into assistant + tool messages
            nlohmann::json j;
            j["role"] = "assistant";
            j["content"] = nullptr;
            if (!msg.reasoning_content.empty()) {
                j["reasoning_content"] = sanitize_utf8(msg.reasoning_content);
            }

            nlohmann::json tc_arr = nlohmann::json::array();
            for (const auto& tc : msg.tool_calls) {
                nlohmann::json tc_json;
                tc_json["id"] = tc.id;
                tc_json["type"] = "function";
                tc_json["function"] = {{"name", tc.name}, {"arguments", tc.arguments}};
                tc_arr.push_back(std::move(tc_json));
            }
            j["tool_calls"] = std::move(tc_arr);
            arr.push_back(std::move(j));
        } else if (msg.role == "tool") {
            nlohmann::json j;
            j["role"] = "tool";
            j["tool_call_id"] = msg.tool_call_id;
            j["content"] = sanitize_utf8(msg.content.value_or(""));
            arr.push_back(std::move(j));
        } else if (msg.role == "user" && (has_multipart_content(msg.parts) || force_multipart)) {
            nlohmann::json j;
            j["role"] = "user";
            if (has_multipart_content(msg.parts)) {
                j["content"] = build_content_array(msg.parts);
            } else {
                nlohmann::json content_arr = nlohmann::json::array();
                content_arr.push_back({{"type", "text"}, {"text", sanitize_utf8(msg.content.value_or(""))}});
                j["content"] = std::move(content_arr);
            }
            arr.push_back(std::move(j));
        } else {
            nlohmann::json j;
            j["role"] = msg.role;
            j["content"] = sanitize_utf8(msg.content.value_or(""));

            if (msg.role == "assistant" && !msg.reasoning_content.empty()) {
                j["reasoning_content"] = sanitize_utf8(msg.reasoning_content);
            }

            arr.push_back(std::move(j));
        }
    }

    return arr;
}

// ---------------------------------------------------------------------------
// make_function_tool — build a function tool definition JSON object
// ---------------------------------------------------------------------------

nlohmann::json make_function_tool(const std::string& name,
                                  const std::string& description,
                                  const nlohmann::json& parameters) {
    return {
        {"type", "function"},
        {"function", {
            {"name", name},
            {"description", description},
            {"parameters", parameters}
        }}
    };
}

// ---------------------------------------------------------------------------
// ToolDef serialization
// ---------------------------------------------------------------------------

void to_json(nlohmann::json& j, const ToolDef& td) {
    j = make_function_tool(td.name, td.description, td.parameters);
}

// ---------------------------------------------------------------------------
// ChatRequest serialization
// ---------------------------------------------------------------------------

nlohmann::json to_json(const ChatRequest& req) {
    // Build messages array
    nlohmann::json arr = nlohmann::json::array();

    // Inject system prompt if set
    if (req.system_prompt.has_value() && !req.system_prompt->empty()) {
        arr.push_back({{"role", "system"}, {"content", sanitize_utf8(*req.system_prompt)}});
    }

    // Check if any user message uses multipart content — forces all user
    // messages to use array form.
    bool force_multipart = false;
    for (const auto& msg : req.messages) {
        if (msg.role == "user" && has_multipart_content(msg.parts)) {
            force_multipart = true;
            break;
        }
    }

    for (const auto& msg : req.messages) {
        if (msg.role == "assistant" && !msg.tool_calls.empty()) {
            // Assistant with tool calls — expand into assistant message
            nlohmann::json j;
            j["role"] = "assistant";
            j["content"] = nullptr;
            if (!msg.reasoning_content.empty()) {
                j["reasoning_content"] = sanitize_utf8(msg.reasoning_content);
            }

            nlohmann::json tc_arr = nlohmann::json::array();
            for (const auto& tc : msg.tool_calls) {
                nlohmann::json tc_json;
                tc_json["id"] = tc.id;
                tc_json["type"] = "function";
                tc_json["function"] = {{"name", tc.name}, {"arguments", tc.arguments}};
                tc_arr.push_back(std::move(tc_json));
            }
            j["tool_calls"] = std::move(tc_arr);
            arr.push_back(std::move(j));
        } else if (msg.role == "tool") {
            nlohmann::json j;
            j["role"] = "tool";
            j["tool_call_id"] = msg.tool_call_id;
            j["content"] = sanitize_utf8(msg.content.value_or(""));
            arr.push_back(std::move(j));
        } else if (msg.role == "user" && (has_multipart_content(msg.parts) || force_multipart)) {
            nlohmann::json j;
            j["role"] = "user";
            if (has_multipart_content(msg.parts)) {
                j["content"] = build_content_array(msg.parts);
            } else {
                nlohmann::json content_arr = nlohmann::json::array();
                content_arr.push_back({{"type", "text"}, {"text", sanitize_utf8(msg.content.value_or(""))}});
                j["content"] = std::move(content_arr);
            }
            arr.push_back(std::move(j));
        } else {
            nlohmann::json j;
            j["role"] = msg.role;
            j["content"] = sanitize_utf8(msg.content.value_or(""));

            if (msg.role == "assistant" && !msg.reasoning_content.empty()) {
                j["reasoning_content"] = sanitize_utf8(msg.reasoning_content);
            }

            arr.push_back(std::move(j));
        }
    }

    // Build tool definitions array
    nlohmann::json tools_arr = nlohmann::json::array();
    for (const auto& td : req.tools) {
        nlohmann::json td_json;
        to_json(td_json, td);
        tools_arr.push_back(std::move(td_json));
    }

    // Construct the full payload
    nlohmann::json payload = {
        {"model", req.model},
        {"messages", std::move(arr)},
        {"stream", req.stream},
        {"stream_options", {{"include_usage", req.include_usage}}}
    };

    // Omit tools key entirely when empty
    if (!tools_arr.empty()) {
        payload["tools"] = std::move(tools_arr);
    }

    // Compute token limit
    int limit = req.max_tokens > 0 ? req.max_tokens
        : (req.context_limit > 0 ? std::min(req.context_limit / 4, 32768) : 32768);

    // Apply thinking/reasoning parameters
    if (req.thinking_enabled) {
        std::string lower = req.model;
        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
        bool is_openai = lower.find("o1") != std::string::npos
                      || lower.find("o3") != std::string::npos
                      || lower.find("o4") != std::string::npos
                      || lower.find("gpt-5") != std::string::npos;
        if (is_openai) {
            // Use reasoning_effort from request if set, otherwise default to "high"
            payload["reasoning_effort"] = req.reasoning_effort.value_or("high");
        } else {
            payload["thinking"] = {{"type", "enabled"}};
        }
        payload["max_completion_tokens"] = limit;
    } else {
        payload["max_tokens"] = limit;
    }

    return payload;
}

} // namespace llmclient
