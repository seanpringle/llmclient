#include "llmclient/types.h"

#include <cstring>
#include <exception>
#include <stdexcept>
#include <string_view>

// ---------------------------------------------------------------------------
// SSEParser
// ---------------------------------------------------------------------------

llmclient::SSEParser::SSEParser(Callbacks cb) : cb_(std::move(cb)) {}

void llmclient::SSEParser::feed(const char* data, size_t len) {
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

void llmclient::SSEParser::process_line(std::string line) {
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

void llmclient::SSEParser::flush() {
    if (!buf_.empty()) {
        process_line(std::move(buf_));
        buf_.clear();
    }
}

void llmclient::SSEParser::reset() {
    buf_.clear();
    raw_.clear();
    current_event_.clear();
}
