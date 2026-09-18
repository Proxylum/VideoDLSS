#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>

namespace dlssvid {

// SHA-256 via Windows CNG (bcrypt). Returns lowercase hex.
std::string Sha256Hex(const void* data, size_t size);
std::string Sha256File(const std::filesystem::path& path);

}  // namespace dlssvid
