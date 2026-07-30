#include "llmclient/types.h"

#include <algorithm>
#include <cstring>
#include <exception>
#include <expected>
#include <set>
#include <stdexcept>
#include <string_view>

namespace llmclient {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Check if a model name matches o-series patterns that require
//  max_completion_tokens on the wire.
static bool is_o_series(const std::string& model) {
    std::string lower = model;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    return lower.find("o1") != std::string::npos
        || lower.find("o3") != std::string::npos
        || lower.find("o4") != std::string::npos
        || lower.find("gpt-5") != std::string::npos;
}

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

    // Parse refusal (safety filter) — mutually exclusive with content
    r.refusal = message.value("refusal", "");

    // content may be null (for tool_call messages or refusal)
    auto content_it = message.find("content");
    if (content_it != message.end() && !content_it->is_null()) {
        if (content_it->is_string()) {
            r.content = content_it->get<std::string>();
        } else if (content_it->is_array()) {
            // Some providers return content as an array of content parts.
            // Concatenate text parts with newlines for backward compatibility.
            std::string combined;
            for (const auto& part : *content_it) {
                if (part.is_object()) {
                    auto type_it = part.find("type");
                    if (type_it != part.end() && type_it->is_string() &&
                        type_it->get<std::string>() == "text") {
                        auto text_it = part.find("text");
                        if (text_it != part.end() && text_it->is_string()) {
                            if (!combined.empty()) combined += '\n';
                            combined += text_it->get<std::string>();
                        }
                    }
                    // Non-text parts (image_url, etc.) are silently skipped.
                }
            }
            r.content = combined;
        }
    }
    r.reasoning_content = message.value("reasoning_content", "");

    // Parse tool_calls — non-streaming has no "index" field, synthesize from position
    // Only include items with type == "function" (skip custom, file_search, etc.)
    auto tc_it = message.find("tool_calls");
    if (tc_it != message.end() && tc_it->is_array()) {
        int idx = 0;
        for (const auto& tc_json : *tc_it) {
            if (tc_json.value("type", "") != "function") {
                continue; // skip non-function tool calls
            }
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
    if (s == "image" || s == "image_url") return ContentPartType::Image;
    return ContentPartType::Text;
}

void to_json(nlohmann::json& j, const ContentPart& cp) {
    if (cp.type == ContentPartType::Text) {
        j = {{"type", "text"}, {"text", cp.text}};
    } else {
        nlohmann::json img;
        img["type"] = "image_url";
        std::string data_url = "data:" + cp.media_type + ";base64," + cp.data;
        img["image_url"]["url"] = data_url;
        if (!cp.detail.empty())
            img["image_url"]["detail"] = cp.detail;
        j = std::move(img);
    }
}

void from_json(const nlohmann::json& j, ContentPart& cp) {
    cp.type = content_part_type_from_str(j.value("type", "text"));
    cp.text = j.value("text", "");
    cp.data = j.value("data", "");
    cp.media_type = j.value("media_type", "");
    cp.detail = j.value("detail", "");
    // Also handle detail nested inside image_url (wire format)
    if (cp.detail.empty()) {
        auto img_it = j.find("image_url");
        if (img_it != j.end() && img_it->is_object()) {
            cp.detail = img_it->value("detail", "");
            // Extract data from the URL if possible
            auto url_it = img_it->find("url");
            if (url_it != img_it->end() && url_it->is_string()) {
                std::string url = url_it->get<std::string>();
                // Parse data: URLs of the form data:<mediatype>;base64,<data>
                auto comma = url.find(',');
                if (comma != std::string::npos) {
                    if (cp.data.empty()) cp.data = url.substr(comma + 1);
                    // Extract media_type from the data URL prefix
                    if (cp.media_type.empty()) {
                        auto semi = url.find(';');
                        if (semi != std::string::npos && semi > 5) {
                            cp.media_type = url.substr(5, semi - 5); // skip "data:"
                        }
                    }
                }
            }
        }
    }
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

void ToolAccumulator::apply(const ToolCallDelta& delta) {
    auto& call = calls_[delta.index];
    call.index = delta.index;

    if (delta.id.has_value()) {
        call.id = delta.id.value();
    }
    if (delta.name.has_value()) {
        call.name = delta.name.value();
    }
    if (delta.arguments_fragment.has_value()) {
        call.arguments += delta.arguments_fragment.value();
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
                // ── Consolidated delta extraction (OpenAI streaming format) ──
                if (j.contains("choices") && j["choices"].is_array() && !j["choices"].empty()) {
                    const auto& choice = j["choices"][0];
                    const auto& delta = choice["delta"];

                    // Build StreamDelta
                    StreamDelta sd;

                    auto extract_str = [&](const std::string& key) -> std::optional<std::string> {
                        auto it = delta.find(key);
                        if (it != delta.end() && it->is_string()) {
                            return it->get<std::string>();
                        }
                        return std::nullopt;
                    };

                    sd.content = extract_str("content");

                    // Reasoning — check both `reasoning_content` and `reasoning`
                    auto r = extract_str("reasoning_content");
                    if (!r.has_value()) {
                        r = extract_str("reasoning");
                    }
                    sd.reasoning_content = std::move(r);

                    // Refusal (safety filter) — may appear in delta
                    sd.refusal = extract_str("refusal");

                    // Finish reason (sibling of delta, not inside delta)
                    auto fr_it = choice.find("finish_reason");
                    if (fr_it != choice.end() && !fr_it->is_null()) {
                        sd.finish_reason = fr_it->get<std::string>();
                    }

                    // Tool calls
                    auto tc_it = delta.find("tool_calls");
                    if (tc_it != delta.end() && tc_it->is_array()) {
                        std::vector<ToolCallDelta> tcs;
                        for (const auto& tc_json : *tc_it) {
                            ToolCallDelta tcd;
                            tcd.index = tc_json.value("index", 0);

                            auto id_it = tc_json.find("id");
                            if (id_it != tc_json.end() && id_it->is_string())
                                tcd.id = id_it->get<std::string>();

                            auto func_it = tc_json.find("function");
                            if (func_it != tc_json.end() && func_it->is_object()) {
                                auto name_it = func_it->find("name");
                                if (name_it != func_it->end() && name_it->is_string())
                                    tcd.name = name_it->get<std::string>();

                                auto args_it = func_it->find("arguments");
                                if (args_it != func_it->end() && args_it->is_string())
                                    tcd.arguments_fragment = args_it->get<std::string>();
                            }

                            tcs.push_back(std::move(tcd));
                        }
                        sd.tool_calls = std::move(tcs);
                    }

                    if (cb_.on_delta) {
                        cb_.on_delta(sd);
                    }
                }

                // Finish reason (sibling of delta, not inside it) — also fire
                // the legacy on_finish callback independently of on_delta.
                if (j.contains("choices") && j["choices"].is_array() && !j["choices"].empty()) {
                    const auto& choice = j["choices"][0];
                    auto fr_it = choice.find("finish_reason");
                    if (fr_it != choice.end() && !fr_it->is_null()) {
                        std::string reason = fr_it->get<std::string>();
                        if (cb_.on_finish) {
                            cb_.on_finish(reason);
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
// ToolDef validation
// ---------------------------------------------------------------------------

static const std::set<std::string> kValidParamTypes = {"string", "integer", "boolean", "number"};

std::expected<void, std::string> ToolDef::validate() const {
    if (name.empty()) {
        return std::unexpected("ToolDef name is empty");
    }

    bool has_params = !params.empty();
    bool has_raw = !raw_schema.empty();

    if (has_params && has_raw) {
        return std::unexpected("ToolDef \"" + name +
                                "\": params and raw_schema are mutually exclusive");
    }

    if (has_raw) {
        // Validate raw_schema is well-formed JSON (schema structure not checked).
        try {
            [[maybe_unused]] auto _ = nlohmann::json::parse(raw_schema);
        } catch (const nlohmann::json::parse_error& e) {
            return std::unexpected("ToolDef \"" + name +
                                    "\": raw_schema is not valid JSON: " + std::string(e.what()));
        }
    }

    if (has_params) {
        std::set<std::string> names;
        for (const auto& p : params) {
            if (p.name.empty()) {
                return std::unexpected("ToolDef \"" + name + "\": ParamDef name is empty");
            }
            if (!names.insert(p.name).second) {
                return std::unexpected("ToolDef \"" + name + "\": duplicate ParamDef name \"" +
                                        p.name + "\"");
            }
            if (!kValidParamTypes.count(p.type)) {
                return std::unexpected("ToolDef \"" + name + "\": ParamDef \"" + p.name +
                                        "\" has invalid type \"" + p.type +
                                        "\" (expected: string, integer, boolean, number)");
            }
        }
    }

    return {};
}

// ---------------------------------------------------------------------------
// ToolDef serialization
// ---------------------------------------------------------------------------

void to_json(nlohmann::json& j, const ToolDef& td) {
    nlohmann::json parameters;

    if (!td.raw_schema.empty()) {
        // Path 2: raw JSON Schema string — parse and embed directly.
        // validate() already verified this is well-formed JSON.
        parameters = nlohmann::json::parse(td.raw_schema);
    } else {
        // Path 1: build the envelope from typed params.
        parameters["type"] = "object";
        nlohmann::json props = nlohmann::json::object();
        nlohmann::json required = nlohmann::json::array();

        for (const auto& p : td.params) {
            props[p.name] = {{"type", p.type}, {"description", p.description}};
            if (p.required) {
                required.push_back(p.name);
            }
        }
        parameters["properties"] = std::move(props);
        parameters["required"] = std::move(required);
    }

    j = {
        {"type", "function"},
        {"function", {
            {"name", td.name},
            {"description", td.description},
            {"parameters", std::move(parameters)}
        }}
    };
}

// ---------------------------------------------------------------------------
// ToolChoice serialization helper
// ---------------------------------------------------------------------------

static void to_json(nlohmann::json& j, const ToolChoice& tc) {
    switch (tc.type) {
        case ToolChoice::Auto:
            j = "auto";
            break;
        case ToolChoice::Required:
            j = "required";
            break;
        case ToolChoice::None:
            j = "none";
            break;
        case ToolChoice::Named:
            j = {{"type", "function"}, {"function", {{"name", tc.name}}}};
            break;
    }
}

// ---------------------------------------------------------------------------
// ResponseFormat serialization helper
// ---------------------------------------------------------------------------

static void to_json(nlohmann::json& j, const ResponseFormat& rf) {
    switch (rf.type) {
        case ResponseFormat::Text:
            j = nullptr; // signals "omit"
            break;
        case ResponseFormat::JsonObject:
            j = {{"type", "json_object"}};
            break;
        case ResponseFormat::JsonSchema:
            j = {
                {"type", "json_schema"},
                {"json_schema", {
                    {"name", "structured_output"},
                    {"schema", nlohmann::json::parse(rf.json_schema)},
                    {"strict", true}
                }}
            };
            break;
    }
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
    };

    // Stream flag — only include stream_options when streaming
    if (req.stream) {
        payload["stream"] = true;
        payload["stream_options"] = {{"include_usage", req.include_usage}};
    }

    // Omit tools key entirely when empty
    if (!tools_arr.empty()) {
        payload["tools"] = std::move(tools_arr);
    }

    // Compute token limit
    int limit = req.max_tokens > 0 ? req.max_tokens
        : (req.context_limit > 0 ? std::min(req.context_limit / 4, 32768) : 32768);

    // Apply thinking/reasoning parameters
    if (req.thinking_enabled) {
        bool openai_o_series = is_o_series(req.model);
        if (openai_o_series) {
            // Use reasoning_effort from request if set, otherwise default to "high"
            payload["reasoning_effort"] = req.reasoning_effort.value_or("high");
        } else {
            payload["thinking"] = {{"type", "enabled"}};
        }
        payload["max_completion_tokens"] = limit;
    } else if (is_o_series(req.model)) {
        // o-series models always use max_completion_tokens even without explicit thinking
        payload["max_completion_tokens"] = limit;
    } else {
        payload["max_tokens"] = limit;
    }

    // ── Tool choice ──
    if (req.tool_choice.has_value()) {
        nlohmann::json tc_json;
        to_json(tc_json, *req.tool_choice);
        payload["tool_choice"] = std::move(tc_json);
    }

    // ── Parallel tool calls ──
    if (!req.parallel_tool_calls && !req.tools.empty()) {
        payload["parallel_tool_calls"] = false;
    }
    // When true (default) and tools are present, omit the field (API default is true).

    // ── Stop sequences ──
    if (!req.stop.empty()) {
        payload["stop"] = req.stop;
    }

    // ── Response format ──
    nlohmann::json rf_json;
    to_json(rf_json, req.response_format);
    if (!rf_json.is_null()) {
        payload["response_format"] = std::move(rf_json);
    }

    return payload;
}

std::string to_json_string(const ChatRequest& req) {
    return to_json(req).dump();
}

} // namespace llmclient
