#pragma once

#include <filesystem>

namespace dlssvid {

// A path that Win32 opens even when it is longer than MAX_PATH (260): absolute paths of 248+ characters get the
// `\\?\` prefix (backslashes only, no `.`/`..`), which bypasses the limit regardless of the system's LongPathsEnabled
// policy or an application manifest. Short and relative paths come back unchanged. The prefixed path is for opening
// and querying files (std::ifstream / std::ofstream / std::filesystem::exists), not for display: log the original.
// Found on the 0.2.0 release check: the TensorRT engine cache name is ~100 characters and an unpacked package in a deep
// folder pushed the file path past 260, so the engine was built and then could not be written.
std::filesystem::path Win32LongPath(const std::filesystem::path& p);

}  // namespace dlssvid
