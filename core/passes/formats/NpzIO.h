#pragma once

#include <filesystem>
#include <map>
#include <string>

#include "passes/PassImage.h"

namespace dlssvid {

// NumPy .npy / .npz (ComfyUI / Depth Anything interchange). Arrays are (H, W) for one
// channel and (H, W, C) otherwise, C-order, little-endian. Written .npz files are stored
// (uncompressed); reading accepts stored and deflate entries (numpy.savez_compressed).
PassImage ReadNpy(const std::filesystem::path& path);
void WriteNpy(const std::filesystem::path& path, const PassImage& img);

// key -> image. WriteNpz stores one array per key.
std::map<std::string, PassImage> ReadNpz(const std::filesystem::path& path);
void WriteNpz(const std::filesystem::path& path, const std::map<std::string, PassImage>& arrays);

// In-memory .npy encoding/decoding (used by the zip layer and by tests).
std::vector<uint8_t> EncodeNpy(const PassImage& img);
PassImage DecodeNpy(const uint8_t* data, size_t size, const std::string& what);

}  // namespace dlssvid
