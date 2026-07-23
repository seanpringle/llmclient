#pragma once

#include <functional>
#include <optional>
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
    int reasoning_tokens = 0;             // from completion_tokens_details (OpenAI)
    int cache_read_input_tokens = 0;      // top-level (Anthropic)
    int cache_creation_input_tokens = 0;   // top-level (Anthropic)
};

void from_json(const nlohmann::json& j, Usage& u);

// ---------------------------------------------------------------------------
// ChatResponse — typed result for a non-streaming chat completion
// ---------------------------------------------------------------------------
// Single-choice assumption: collapses to choices[0].

struct ChatResponse {
    std::string content;
    std::string reasoning_content;
    std::vector<ToolCall> tool_calls;
    Usage usage;
    std::string finish_reason; // "stop", "tool_calls", "length", etc.
    std::string id;
    std::string model;
};

void from_json(const nlohmann::json& j, ChatResponse& r);

// ---------------------------------------------------------------------------
// ToolCallDelta — a single streaming tool-call delta fragment
// ---------------------------------------------------------------------------

struct ToolCallDelta {
    int index;
    std::optional<std::string> id;               // set on first fragment for this index
    std::optional<std::string> name;             // set on first fragment for this index
    std::optional<std::string> arguments_fragment; // partial args, appended to accumulated string
};

// ---------------------------------------------------------------------------
// StreamDelta — consolidated delta for one streaming chunk
// ---------------------------------------------------------------------------
// finish_reason is NOT here — delivered via SSEParser::Callbacks::on_finish.

struct StreamDelta {
    std::optional<std::string> content;
    std::optional<std::string> reasoning_content;
    std::optional<std::vector<ToolCallDelta>> tool_calls;
};

// ---------------------------------------------------------------------------
// ToolAccumulator — merges streaming tool_call deltas across SSE chunks
// ---------------------------------------------------------------------------

class ToolAccumulator {
  public:
    void apply(const ToolCallDelta& delta);
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
    using DoneCallback = std::function<void()>;
    using ErrorCallback = std::function<void(const std::string&)>;

    struct Callbacks {
        DoneCallback on_done;
        ErrorCallback on_error;

        /// Called once per chunk with the consolidated delta.
        std::function<void(const StreamDelta& delta)> on_delta;

        /// Streaming finish_reason — fires on the final chunk (before on_done).
        std::function<void(std::string_view reason)> on_finish;

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

// ---------------------------------------------------------------------------
// ToolDef — strongly-typed tool definition (replaces raw make_function_tool)
// ---------------------------------------------------------------------------

struct ToolDef {
    std::string name;
    std::string description;
    nlohmann::json parameters;     // JSON Schema — inherently flexible
};

void to_json(nlohmann::json& j, const ToolDef& td);

// ---------------------------------------------------------------------------
// ChatRequest — typed request payload for chat completions
// ---------------------------------------------------------------------------

struct ChatRequest {
    std::string model;
    std::vector<ProtocolMessage> messages;
    std::vector<ToolDef> tools;
    bool stream = false;

    // ── Token limits ──
    // Max tokens hint. The library decides which wire field to write:
    //   max_completion_tokens (when thinking_enabled=true) or
    //   max_tokens (thinking_enabled=false).
    // If 0, falls back to context_limit/4 (capped at 32768).
    int max_tokens = 0;

    // Context window size from API metadata. Used only when max_tokens==0
    // to derive the fallback limit. 0 = use hardcoded default (32768).
    int context_limit = 0;

    // ── Thinking / Reasoning ──
    bool thinking_enabled = false;
    // Optional override for the reasoning_effort field sent to the API.
    // When unset, the library applies its built-in model detection
    // (e.g. "high" for OpenAI o-series, nothing for non-OpenAI thinking models).
    std::optional<std::string> reasoning_effort;

    // ── System prompt ──
    // Injected as the first system-role message in the messages array.
    // If absent, no system message is prepended.
    std::optional<std::string> system_prompt;

    // ── Streaming options ──
    bool include_usage = true;        // maps to stream_options.include_usage
};

/// Serialize a ChatRequest to an OpenAI-compatible JSON payload.
nlohmann::json to_json(const ChatRequest& req);

} // namespace llmclient
