#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace dlssvid {

// Child process with line-oriented stdin/stdout pipes (Windows). stderr is inherited so the
// child's logging shows up in ours.
class Subprocess {
public:
    Subprocess() = default;
    ~Subprocess();
    Subprocess(const Subprocess&) = delete;
    Subprocess& operator=(const Subprocess&) = delete;

    // `args[0]` is the executable. Throws dlssvid::Error if the process cannot start.
    void Start(const std::vector<std::string>& args, const std::filesystem::path& workingDir = {});
    bool Running() const;
    std::optional<uint32_t> ExitCode() const;  // set once the process has exited

    void WriteLine(const std::string& line);
    // Blocks until a full line arrives; nullopt at EOF (child exited / closed stdout).
    std::optional<std::string> ReadLine(uint32_t timeoutMs = 0xFFFFFFFFu);
    void CloseStdin();
    void Terminate();
    uint32_t Wait(uint32_t timeoutMs = 0xFFFFFFFFu);

    // Run to completion, capturing stdout. Returns exit code.
    static uint32_t Run(const std::vector<std::string>& args, std::string* stdoutText = nullptr, const std::filesystem::path& workingDir = {});

private:
    void* process_ = nullptr;
    void* stdinWrite_ = nullptr;
    void* stdoutRead_ = nullptr;
    std::string buffer_;
    bool eof_ = false;
};

std::string QuoteArg(const std::string& arg);  // Windows command-line quoting

}  // namespace dlssvid
