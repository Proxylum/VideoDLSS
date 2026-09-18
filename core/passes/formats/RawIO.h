#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "passes/PassImage.h"

namespace dlssvid {

// Headerless little-endian dumps (.r32, .rg16f, ...): the DLSS input as it is fed to NGX.
// Geometry comes from the manifest.
PassImage ReadRaw(const std::filesystem::path& path, uint32_t width, uint32_t height, PixelType type,
                  const std::vector<std::string>& channels);
void WriteRaw(const std::filesystem::path& path, const PassImage& img);

}  // namespace dlssvid
