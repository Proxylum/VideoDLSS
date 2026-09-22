#include "util/Paths.h"

#include <string>

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

}  // namespace dlssvid
