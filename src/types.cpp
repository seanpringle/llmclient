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

} // namespace llmclient
