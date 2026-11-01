#include "util/Paths.h"

#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#include <shellapi.h>
#endif

#include "util/Error.h"

namespace dlssvid {

std::filesystem::path Win32LongPath(const std::filesystem::path& p) {
#ifdef _WIN32
    constexpr size_t kPrefixFrom = 248;  // MAX_PATH minus room for a file name: prefix before Win32 refuses
    if (p.empty() || !p.is_absolute()) return p;
    std::wstring s = p.lexically_normal().wstring();
    if (s.rfind(L"\\\\?\\", 0) == 0) return p;
    if (s.size() < kPrefixFrom) return p;
    for (wchar_t& c : s)
        if (c == L'/') c = L'\\';
    if (s.rfind(L"\\\\", 0) == 0) return std::filesystem::path(L"\\\\?\\UNC\\" + s.substr(2));  // \\server\share -> \\?\UNC\server\share
    return std::filesystem::path(L"\\\\?\\" + s);
#else
    return p;
#endif
}

std::filesystem::path CurrentExecutable() {
#ifdef _WIN32
    std::vector<wchar_t> buf(32768);
    const DWORD n = GetModuleFileNameW(nullptr, buf.data(), static_cast<DWORD>(buf.size()));
    if (n == 0 || n >= buf.size()) Throw("cannot get the path of the running executable (error " + std::to_string(GetLastError()) + ")");
    return std::filesystem::path(std::wstring(buf.data(), n));
#else
    return std::filesystem::read_symlink("/proc/self/exe");
#endif
}

bool Utf8CodePage() {
#ifdef _WIN32
    return GetACP() == CP_UTF8;
#else
    return true;
#endif
}

int FirstNonAsciiArgument(const wchar_t* commandLine) {
#ifdef _WIN32
    int n = 0;
    LPWSTR* args = commandLine && *commandLine ? CommandLineToArgvW(commandLine, &n) : nullptr;
    if (!args) return 0;
    int found = 0;
    for (int i = 1; i < n && !found; ++i)
        for (const wchar_t* c = args[i]; *c && !found; ++c)
            if (*c >= 0x80) found = i;
    LocalFree(args);
    return found;
#else
    (void)commandLine;
    return 0;
#endif
}

int FirstNonAsciiArgument() {
#ifdef _WIN32
    return FirstNonAsciiArgument(GetCommandLineW());
#else
    return 0;
#endif
}

}  // namespace dlssvid
