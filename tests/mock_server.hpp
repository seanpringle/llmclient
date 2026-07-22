#pragma once

#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <functional>
#include <string>
#include <thread>

// -------------------------------------------------------------------
// Minimal HTTP mock server for testing chat completions
// -------------------------------------------------------------------
// Handler returns the response body.  If streaming=true the Content-Type
// is set to text/event-stream; otherwise application/json with Content-Length.
// Binds to a random port (port 0).  Port is available via port() / base_url().
// Can return non-200 status codes via the second element of the pair.
// -------------------------------------------------------------------

class MockServer {
public:
    using Handler =
        std::function<std::string(const std::string& request)>;

    MockServer(Handler handler, bool streaming = false, int status_code = 200)
        : handler_(std::move(handler)), streaming_(streaming), status_code_(status_code) {
        server_fd_ = socket(AF_INET, SOCK_STREAM, 0);

        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = INADDR_ANY;
        addr.sin_port = 0;

        bind(server_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
        listen(server_fd_, 5);

        socklen_t len = sizeof(addr);
        getsockname(server_fd_, reinterpret_cast<sockaddr*>(&addr), &len);
        port_ = ntohs(addr.sin_port);

        running_ = true;
        thread_ = std::jthread([this] { run(); });
    }

    ~MockServer() {
        running_ = false;
        if (server_fd_ >= 0) {
            close(server_fd_);
            server_fd_ = -1;
        }
    }

    int port() const { return port_; }
    std::string base_url() const {
        return "http://127.0.0.1:" + std::to_string(port_) + "/v1";
    }

private:
    void run() {
        while (running_) {
            struct pollfd pfd = {server_fd_, POLLIN, 0};
            int ret = poll(&pfd, 1, 200);
            if (!running_) break;
            if (ret <= 0) continue;

            sockaddr_in client_addr{};
            socklen_t client_len = sizeof(client_addr);
            int client =
                accept(server_fd_, reinterpret_cast<sockaddr*>(&client_addr),
                       &client_len);
            if (client < 0)
                continue;

            std::vector<char> buf(65536, 0);
            // Read request headers + body.  Parse Content-Length to
            // determine how many body bytes we still need.
            ssize_t n = read(client, buf.data(), buf.size() - 1);
            if (n > 0) {
                buf[n] = '\0';
                std::string req(buf.data(), n);

                // Find Content-Length
                long body_len = 0;
                auto cl_pos = req.find("Content-Length: ");
                if (cl_pos != std::string::npos) {
                    char* end = nullptr;
                    body_len = std::strtol(
                        req.c_str() + cl_pos + 16, &end, 10);
                }
                // Find start of body (after \r\n\r\n)
                auto hdr_end = req.find("\r\n\r\n");
                if (hdr_end != std::string::npos) {
                    hdr_end += 4;  // skip past \r\n\r\n
                    long have = static_cast<long>(n) -
                                static_cast<long>(hdr_end);
                    while (have < body_len &&
                           static_cast<size_t>(n) < buf.size() - 1) {
                        ssize_t more =
                            read(client, buf.data() + n, buf.size() - 1 - n);
                        if (more <= 0)
                            break;
                        n += more;
                        have += more;
                    }
                }

                buf[n] = '\0';
                std::string request(buf.data(), n);
                std::string body = handler_(request);

                std::string status_line = "HTTP/1.1 " + std::to_string(status_code_) + " ";
                if (status_code_ == 200) status_line += "OK";
                else if (status_code_ == 400) status_line += "Bad Request";
                else if (status_code_ == 429) status_line += "Too Many Requests";
                else if (status_code_ == 503) status_line += "Service Unavailable";
                else status_line += "Unknown";

                std::string response;
                if (streaming_) {
                    response =
                        status_line + "\r\n"
                        "Content-Type: text/event-stream\r\n"
                        "Cache-Control: no-cache\r\n"
                        "\r\n" +
                        body;
                } else {
                    response =
                        status_line + "\r\n"
                        "Content-Type: application/json\r\n"
                        "Content-Length: " +
                        std::to_string(body.size()) + "\r\n" + "\r\n" + body;
                }
                write(client, response.data(), response.size());
                // flush + ensure data is transmitted before close
                shutdown(client, SHUT_WR);
                char discard[1024];
                while (read(client, discard, sizeof(discard)) > 0) {}
            }
            close(client);
        }
    }

    int server_fd_ = -1;
    int port_ = 0;
    std::atomic<bool> running_{false};
    std::jthread thread_;
    Handler handler_;
    bool streaming_ = false;
    int status_code_ = 200;
};
