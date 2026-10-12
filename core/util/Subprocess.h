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

    // `args[0]` is the executable; arguments are UTF-8 (pass paths through Utf8Arg). Throws dlssvid::Error if the process
    // cannot start. killWithOwner: the child runs in a job object that ends it when this object or our process goes away,
    // and that suppresses the Windows Error Reporting dialog for its unhandled exceptions - a crashed child exits instead
    // of hanging on a dialog. Only the child itself: processes it starts leave the job silently (SILENT_BREAKAWAY - so
    // that NGX's nvngx_update.exe lives on as it does without a job), so a child with workers of its own must end them.
    void Start(const std::vector<std::string>& args, const std::filesystem::path& workingDir = {}, bool killWithOwner = false);
    bool Running() const;
    uint32_t Pid() const;
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
    void* job_ = nullptr;
    void* stdinWrite_ = nullptr;
    void* stdoutRead_ = nullptr;
    std::string buffer_;
    bool eof_ = false;
};

std::string QuoteArg(const std::string& arg);  // Windows command-line quoting
std::string Utf8Arg(const std::filesystem::path& p);  // a path as the UTF-8 argument Subprocess::Start expects (not path.string(): ACP)

}  // namespace dlssvid
