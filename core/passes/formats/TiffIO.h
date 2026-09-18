#pragma once

#include <filesystem>

#include "passes/PassImage.h"

namespace dlssvid {

// TIFF: u8 / u16 / f16 / f32, 1-4 contiguous channels, deflate compression, lossless.
PassImage ReadTiff(const std::filesystem::path& path);
void WriteTiff(const std::filesystem::path& path, const PassImage& img);

}  // namespace dlssvid
