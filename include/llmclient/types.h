#pragma once

#include <functional>
#include <string>

#include <nlohmann/json.hpp>

namespace llmclient {

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

} // namespace llmclient
