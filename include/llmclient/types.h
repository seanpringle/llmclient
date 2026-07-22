#pragma once

#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

namespace llmclient {

// ---------------------------------------------------------------------------
// ContentPartType — type of a multipart content part
// ---------------------------------------------------------------------------

enum class ContentPartType { Text, Image };

// ---------------------------------------------------------------------------
// ContentPart — a single part of a multipart message (protocol only)
// ---------------------------------------------------------------------------

struct ContentPart {
    ContentPartType type = ContentPartType::Text;
    std::string text;          // for Text parts
    std::string data;          // base64 data
    std::string media_type;    // e.g. "image/jpeg", "image/png"
    std::string detail;        // OpenAI detail param: "auto", "low", "high"
};

void to_json(nlohmann::json& j, const ContentPart& cp);
void from_json(const nlohmann::json& j, ContentPart& cp);

/// Build an OpenAI content array from a vector of ContentParts.
nlohmann::json build_content_array(const std::vector<ContentPart>& parts);

/// Returns true if any ContentPart in the vector is non-text.
bool has_multipart_content(const std::vector<ContentPart>& parts);

/// Returns true if any message in the vector uses multipart content.
bool any_user_multipart(const std::vector<std::string>& roles,
                        const std::vector<std::vector<ContentPart>>& parts_list);

// ---------------------------------------------------------------------------
// ToolCall — represents a function call requested by the model (protocol only)
// ---------------------------------------------------------------------------

struct ToolCall {
    int index = 0;
    std::string id;
    std::string name;
    std::string arguments; // accumulated JSON fragment from streaming
    // NOTE: tool execution result is NOT part of this protocol struct.
    // The consuming application stores results separately.
};

// ---------------------------------------------------------------------------
// Usage — token usage reported by the API
// ---------------------------------------------------------------------------

struct Usage {
    int prompt_tokens = 0;
    int completion_tokens = 0;
    int total_tokens = 0;
};

void from_json(const nlohmann::json& j, Usage& u);

// ---------------------------------------------------------------------------
// ToolAccumulator — merges streaming tool_call deltas across SSE chunks
// ---------------------------------------------------------------------------

class ToolAccumulator {
  public:
    void apply(const nlohmann::json& delta);
    bool has_calls() const { return !calls_.empty(); }
    std::vector<ToolCall> finalize() const;

  private:
    std::unordered_map<int, ToolCall> calls_;
};

// ---------------------------------------------------------------------------
// SSEParser — incremental SSE line parser for curl write callback
// ---------------------------------------------------------------------------

class SSEParser {
  public:
    using DataCallback = std::function<void(const std::string& event, const nlohmann::json& data)>;
    using DoneCallback = std::function<void()>;
    using ErrorCallback = std::function<void(const std::string&)>;

    struct Callbacks {
        DataCallback on_data;
        DoneCallback on_done;
        ErrorCallback on_error;

        // ── Structured streaming callbacks (optional, OpenAI delta format) ──
        // These are called in addition to on_data when the JSON carries the
        // corresponding field.  They save consumers from manually extracting
        // fields from the delta JSON.

        /// Called for each content delta chunk.
        std::function<void(std::string_view text)> on_content_delta;
        /// Called for reasoning content delta (both `reasoning_content` and
        /// `reasoning` field names are checked).
        std::function<void(std::string_view text)> on_reasoning_delta;
        /// Called for each tool_calls delta fragment (raw JSON array element).
        std::function<void(const nlohmann::json& delta)> on_tool_call_delta;
        /// Called once with the final usage object.
        std::function<void(Usage u)> on_usage;
    };

    explicit SSEParser(Callbacks cb);
    void feed(const char* data, size_t len);
    void flush();
    void reset();
    const std::string& raw() const { return raw_; }

  private:
    void process_line(std::string line);

    Callbacks cb_;
    std::string buf_;
    std::string raw_;
    std::string current_event_;
};

// ---------------------------------------------------------------------------
// sanitize_utf8 — small utility to ensure valid UTF-8 output
// ---------------------------------------------------------------------------

std::string sanitize_utf8(const std::string& input);

// ---------------------------------------------------------------------------
// ProtocolMessage — one message in the API conversation (wire format)
// ---------------------------------------------------------------------------

struct ProtocolMessage {
    std::string role;                             // system, user, assistant, tool
    std::optional<std::string> content;           // nullopt for tool_call messages
    std::vector<ContentPart> parts;               // multipart content (user images)
    std::vector<ToolCall> tool_calls;             // for assistant tool_call msgs
    std::string tool_call_id;                     // for tool result messages
    std::string reasoning_content;                // model-specific, may be empty
};

/// Build the OpenAI-compatible messages array from a vector of ProtocolMessages.
nlohmann::json build_openai_payload(const std::vector<ProtocolMessage>& messages,
                                    const std::string& system_prompt);

/// Build a function tool definition JSON object.
nlohmann::json make_function_tool(const std::string& name,
                                  const std::string& description,
                                  const nlohmann::json& parameters);

} // namespace llmclient
