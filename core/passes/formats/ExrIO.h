#pragma once

#include <filesystem>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "passes/PassImage.h"

namespace dlssvid {

// OpenEXR: F16 / F32 channels stored losslessly (ZIP compression). U8/U16 images are
// written as F16/F32 (value-preserving) and read back as float.
PassImage ReadExr(const std::filesystem::path& path);
void WriteExr(const std::filesystem::path& path, const PassImage& img);

// Multi-layer EXR (Nuke / Resolve preset): each layer's channels are stored as
// "<layer>.<channel>" (empty layer = top-level "R","G","B").
using ExrLayers = std::vector<std::pair<std::string, PassImage>>;
void WriteExrLayers(const std::filesystem::path& path, const ExrLayers& layers);
std::map<std::string, PassImage> ReadExrLayers(const std::filesystem::path& path);

}  // namespace dlssvid
