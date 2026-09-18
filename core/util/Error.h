#pragma once

#include <stdexcept>
#include <string>

namespace dlssvid {

class Error : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

[[noreturn]] void Throw(const std::string& message);

// HRESULT helpers (HRESULT is a LONG; avoid pulling <windows.h> into every header).
std::string HrToString(long hr);
void CheckHr(long hr, const char* what);

// FFmpeg error helpers.
std::string AvErrorToString(int err);
void CheckAv(int err, const char* what);

}  // namespace dlssvid
