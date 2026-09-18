#pragma once

#include <filesystem>
#include <optional>

#include "passes/Manifest.h"
#include "passes/PassImage.h"

namespace dlssvid {

// Format dispatch for one pass frame file. The manifest supplies geometry for raw dumps and
// lets the reader restore canonical channel names / sample types.
PassImage ReadPassFile(const std::filesystem::path& path, const Manifest* manifest = nullptr);

// Writes `img` in the format implied by the extension of `path`. Integer/float conversions
// needed by the container (e.g. PNG needs u8/u16) must be done by the caller; this throws.
void WritePassFile(const std::filesystem::path& path, const PassImage& img);

// Converts an image to the sample type a format can store for a given pass, e.g. colour ->
// u16 for PNG, depth -> f32 for EXR. Value-preserving where the format allows it.
PassImage PrepareForFormat(const PassImage& img, FileFormat format, PixelType wanted);

}  // namespace dlssvid
