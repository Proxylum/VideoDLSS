#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "passes/PassImage.h"

namespace dlssvid {

// The header of an image file without its pixels (PR-8: the control mask's import refuses a file too large or of a kind it cannot
// read before anything is decoded or allocated). The kind comes from the signature, not the extension: a ".png" that is a TIFF is
// read as the TIFF it is.
struct ImageProbe {
    enum class Format : uint8_t { Unknown, Png, Tiff, Exr } format = Format::Unknown;
    uint32_t width = 0, height = 0;  // EXR: the display window
    uint32_t channels = 0;           // what the reader returns: PNG after its expansions (palette -> RGB, tRNS -> alpha); EXR: the layer
                                     // ReadExr returns (the top-level one, else the first by name)
    PixelType type = PixelType::U8;  // the reader's sample type (EXR: of that layer - F32 when a channel of it is FLOAT / UINT, else F16)
    // TIFF: photometric (MINISWHITE 0, MINISBLACK 1, RGB 2, PALETTE 3, ...; a missing tag is guessed from the samples, as libtiff
    // does), tiles, separate planes, whether ReadTiff takes the samples (1-4 of 8 / 16 bit unsigned, 16 / 32 bit float) and whether this
    // build of libtiff decodes the compression (JPEG is not built in)
    uint16_t tiffPhotometric = 0, tiffCompression = 1;
    bool tiffTiled = false, tiffPlanarSeparate = false, tiffSamplesSupported = true, tiffCodecSupported = true;
    std::vector<std::string> exrChannels;  // every channel of an EXR, layers included
    // EXR: the data window relative to the display window's origin (ReadExr returns the data window)
    int32_t exrDataMinX = 0, exrDataMinY = 0;
    uint32_t exrDataWidth = 0, exrDataHeight = 0;
};

// PNG 89 50 4E 47, TIFF "II*\0" / "MM\0*", EXR 76 2F 31 01; Unknown otherwise (a file shorter than 4 bytes too). Throws when the file
// does not open or its first bytes do not read.
ImageProbe::Format ImageFormatOf(const std::filesystem::path& path);
std::string_view ToString(ImageProbe::Format f);  // png | tiff | exr | unknown
// The header; Format::Unknown for another kind of file. Throws when the header of a PNG, TIFF or EXR does not read.
ImageProbe ProbeImage(const std::filesystem::path& path);

}  // namespace dlssvid
