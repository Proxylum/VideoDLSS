#pragma once

#include <filesystem>

#include "passes/PassImage.h"

namespace dlssvid {

// PNG 8/16-bit, 1-4 channels (gray, gray+alpha, RGB, RGBA). Float images are rejected:
// callers convert explicitly (see PassSequence) so that no silent quantisation happens.
PassImage ReadPng(const std::filesystem::path& path);
void WritePng(const std::filesystem::path& path, const PassImage& img);

}  // namespace dlssvid
