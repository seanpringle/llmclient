#pragma once

#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

namespace llmclient {

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

} // namespace llmclient
