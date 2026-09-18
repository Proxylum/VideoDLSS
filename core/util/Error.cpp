#include "util/Error.h"

#include <windows.h>

#include <cstdio>

extern "C" {
#include <libavutil/error.h>
}

namespace dlssvid {

void Throw(const std::string& message) { throw Error(message); }

std::string HrToString(long hr) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "0x%08lX", static_cast<unsigned long>(hr));
    std::string text = buf;
    char* sys = nullptr;
    const DWORD n = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                                   nullptr, static_cast<DWORD>(hr), MAKELANGID(LANG_ENGLISH, SUBLANG_ENGLISH_US),
                                   reinterpret_cast<LPSTR>(&sys), 0, nullptr);
    if (n && sys) {
        std::string msg(sys, n);
        while (!msg.empty() && (msg.back() == '\r' || msg.back() == '\n' || msg.back() == ' ')) msg.pop_back();
        text += " (" + msg + ")";
    }
    if (sys) LocalFree(sys);
    return text;
}

void CheckHr(long hr, const char* what) {
    if (FAILED(hr)) Throw(std::string(what) + " failed: HRESULT " + HrToString(hr));
}

std::string AvErrorToString(int err) {
    char buf[AV_ERROR_MAX_STRING_SIZE] = {};
    av_strerror(err, buf, sizeof(buf));
    return buf;
}

void CheckAv(int err, const char* what) {
    if (err < 0) Throw(std::string(what) + " failed: " + AvErrorToString(err));
}

}  // namespace dlssvid
