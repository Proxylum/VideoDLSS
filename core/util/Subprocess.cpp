#include "util/Subprocess.h"

#include <windows.h>

#include "util/Error.h"

namespace dlssvid {

std::string QuoteArg(const std::string& arg) {
    if (!arg.empty() && arg.find_first_of(" \t\"") == std::string::npos) return arg;
    std::string out = "\"";
    unsigned backslashes = 0;
    for (char c : arg) {
        if (c == '\\') {
            ++backslashes;
        } else if (c == '"') {
            out.append(backslashes * 2 + 1, '\\');
            out.push_back('"');
            backslashes = 0;
        } else {
            out.append(backslashes, '\\');
            out.push_back(c);
            backslashes = 0;
        }
    }
    out.append(backslashes * 2, '\\');
    out.push_back('"');
    return out;
}

namespace {
std::wstring Widen(const std::string& s) {
    if (s.empty()) return {};
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
    std::wstring w(static_cast<size_t>(n > 0 ? n - 1 : 0), L'\0');
    if (n > 1) MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, w.data(), n);
    return w;
}
}  // namespace

Subprocess::~Subprocess() {
    if (stdinWrite_) CloseHandle(stdinWrite_);
    if (stdoutRead_) CloseHandle(stdoutRead_);
    if (process_) {
        if (WaitForSingleObject(process_, 0) == WAIT_TIMEOUT) TerminateProcess(process_, 1);
        CloseHandle(process_);
    }
}

void Subprocess::Start(const std::vector<std::string>& args, const std::filesystem::path& workingDir) {
    if (args.empty()) Throw("Subprocess::Start: no executable");
    SECURITY_ATTRIBUTES sa{sizeof(sa), nullptr, TRUE};
    HANDLE inRead = nullptr, inWrite = nullptr, outRead = nullptr, outWrite = nullptr;
    if (!CreatePipe(&inRead, &inWrite, &sa, 0) || !CreatePipe(&outRead, &outWrite, &sa, 0)) Throw("CreatePipe failed");
    SetHandleInformation(inWrite, HANDLE_FLAG_INHERIT, 0);
    SetHandleInformation(outRead, HANDLE_FLAG_INHERIT, 0);

    std::string cmd;
    for (size_t i = 0; i < args.size(); ++i) cmd += (i ? " " : "") + QuoteArg(args[i]);
    std::wstring wcmd = Widen(cmd);
    const std::wstring wdir = workingDir.empty() ? std::wstring() : workingDir.wstring();

    STARTUPINFOW si{};
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdInput = inRead;
    si.hStdOutput = outWrite;
    si.hStdError = GetStdHandle(STD_ERROR_HANDLE);
    PROCESS_INFORMATION pi{};
    const BOOL ok = CreateProcessW(nullptr, wcmd.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW, nullptr,
                                   wdir.empty() ? nullptr : wdir.c_str(), &si, &pi);
    CloseHandle(inRead);
    CloseHandle(outWrite);
    if (!ok) {
        CloseHandle(inWrite);
        CloseHandle(outRead);
        Throw("cannot start process: " + cmd + " (error " + std::to_string(GetLastError()) + ")");
    }
    CloseHandle(pi.hThread);
    process_ = pi.hProcess;
    stdinWrite_ = inWrite;
    stdoutRead_ = outRead;
}

bool Subprocess::Running() const { return process_ && WaitForSingleObject(process_, 0) == WAIT_TIMEOUT; }

std::optional<uint32_t> Subprocess::ExitCode() const {
    if (!process_ || Running()) return std::nullopt;
    DWORD code = 0;
    GetExitCodeProcess(process_, &code);
    return code;
}

void Subprocess::WriteLine(const std::string& line) {
    if (!stdinWrite_) Throw("Subprocess::WriteLine: stdin closed");
    const std::string data = line + "\n";
    DWORD written = 0;
    if (!WriteFile(stdinWrite_, data.data(), static_cast<DWORD>(data.size()), &written, nullptr) || written != data.size())
        Throw("Subprocess::WriteLine failed (child exited?)");
}

std::optional<std::string> Subprocess::ReadLine(uint32_t timeoutMs) {
    const ULONGLONG start = GetTickCount64();
    for (;;) {
        const size_t nl = buffer_.find('\n');
        if (nl != std::string::npos) {
            std::string line = buffer_.substr(0, nl);
            buffer_.erase(0, nl + 1);
            if (!line.empty() && line.back() == '\r') line.pop_back();
            return line;
        }
        if (eof_) {
            if (buffer_.empty()) return std::nullopt;
            std::string line = std::move(buffer_);
            buffer_.clear();
            return line;
        }
        DWORD avail = 0;
        if (!PeekNamedPipe(stdoutRead_, nullptr, 0, nullptr, &avail, nullptr)) {
            eof_ = true;
            continue;
        }
        if (avail == 0) {
            if (!Running()) {
                // drain anything written just before exit
                if (!PeekNamedPipe(stdoutRead_, nullptr, 0, nullptr, &avail, nullptr) || avail == 0) {
                    eof_ = true;
                    continue;
                }
            } else {
                if (timeoutMs != 0xFFFFFFFFu && GetTickCount64() - start > timeoutMs) return std::nullopt;
                Sleep(2);
                continue;
            }
        }
        char buf[4096];
        DWORD read = 0;
        if (!ReadFile(stdoutRead_, buf, static_cast<DWORD>(std::min<DWORD>(avail, sizeof(buf))), &read, nullptr) || read == 0) {
            eof_ = true;
            continue;
        }
        buffer_.append(buf, read);
    }
}

void Subprocess::CloseStdin() {
    if (stdinWrite_) CloseHandle(stdinWrite_);
    stdinWrite_ = nullptr;
}

void Subprocess::Terminate() {
    if (process_ && Running()) TerminateProcess(process_, 1);
}

uint32_t Subprocess::Wait(uint32_t timeoutMs) {
    if (!process_) return 0;
    WaitForSingleObject(process_, timeoutMs);
    DWORD code = 0;
    GetExitCodeProcess(process_, &code);
    return code;
}

uint32_t Subprocess::Run(const std::vector<std::string>& args, std::string* stdoutText, const std::filesystem::path& workingDir) {
    Subprocess p;
    p.Start(args, workingDir);
    p.CloseStdin();
    std::string all;
    while (auto line = p.ReadLine()) all += *line + "\n";
    if (stdoutText) *stdoutText = all;
    return p.Wait();
}

}  // namespace dlssvid
