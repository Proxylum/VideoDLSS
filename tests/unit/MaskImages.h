#pragma once

// Image files the readers of the core do not write (PR-8, the control mask's import): TIFF with any photometric, tiles or separate
// planes, EXR whose data window is not its display window, and a PNG whose header promises a size its data does not hold.

#include <OpenEXR/ImfChannelList.h>
#include <OpenEXR/ImfFrameBuffer.h>
#include <OpenEXR/ImfHeader.h>
#include <OpenEXR/ImfOutputFile.h>
#include <tiffio.h>
#include <zlib.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "passes/formats/ExrIO.h"
#include "util/Error.h"
#include "util/Half.h"

namespace dlssvid::test {

struct TiffSpec {
    uint32_t width = 8, height = 4;
    uint16_t samples = 1, bits = 8, sampleFormat = SAMPLEFORMAT_UINT, photometric = PHOTOMETRIC_MINISBLACK;
    bool tiled = false, planarSeparate = false;
};
// value(x, y, sample): the code (integers) or the float stored
inline void WriteTiffWith(const std::filesystem::path& path, const TiffSpec& s, const std::function<double(uint32_t, uint32_t, uint32_t)>& value) {
    std::unique_ptr<TIFF, void (*)(TIFF*)> tif(TIFFOpenW(path.wstring().c_str(), "w"), &TIFFClose);
    if (!tif) Throw("cannot create TIFF " + path.string());
    TIFFSetField(tif.get(), TIFFTAG_IMAGEWIDTH, s.width);
    TIFFSetField(tif.get(), TIFFTAG_IMAGELENGTH, s.height);
    TIFFSetField(tif.get(), TIFFTAG_BITSPERSAMPLE, s.bits);
    TIFFSetField(tif.get(), TIFFTAG_SAMPLESPERPIXEL, s.samples);
    TIFFSetField(tif.get(), TIFFTAG_SAMPLEFORMAT, s.sampleFormat);
    TIFFSetField(tif.get(), TIFFTAG_PHOTOMETRIC, s.photometric);
    TIFFSetField(tif.get(), TIFFTAG_PLANARCONFIG, s.planarSeparate ? PLANARCONFIG_SEPARATE : PLANARCONFIG_CONTIG);
    if (const uint16_t colour = s.photometric == PHOTOMETRIC_RGB ? 3 : 1; s.samples > colour) {  // Photoshop: transparency, saved selections
        std::vector<uint16_t> extra(s.samples - colour, static_cast<uint16_t>(EXTRASAMPLE_UNSPECIFIED));
        extra[0] = EXTRASAMPLE_UNASSALPHA;
        TIFFSetField(tif.get(), TIFFTAG_EXTRASAMPLES, static_cast<uint16_t>(extra.size()), extra.data());
    }
    if (s.photometric == PHOTOMETRIC_PALETTE) {
        std::vector<uint16_t> map(size_t{1} << s.bits);
        for (size_t i = 0; i < map.size(); ++i) map[i] = static_cast<uint16_t>(i * 257);
        TIFFSetField(tif.get(), TIFFTAG_COLORMAP, map.data(), map.data(), map.data());
    }
    if (s.bits < 8) {  // bilevel / packed: zero rows of the packed size (the import refuses them by the header anyway)
        TIFFSetField(tif.get(), TIFFTAG_ROWSPERSTRIP, s.height);
        std::vector<uint8_t> row((static_cast<size_t>(s.width) * s.samples * s.bits + 7) / 8, uint8_t{0});
        for (uint32_t y = 0; y < s.height; ++y) TIFFWriteScanline(tif.get(), row.data(), y, 0);
        return;
    }
    const size_t bytes = s.bits / 8;
    auto store = [&](uint8_t* dst, double v) {
        if (s.sampleFormat == SAMPLEFORMAT_IEEEFP && s.bits == 32) {
            const float f = static_cast<float>(v);
            std::memcpy(dst, &f, 4);
        } else if (s.sampleFormat == SAMPLEFORMAT_IEEEFP && s.bits == 16) {
            const uint16_t h = FloatToHalf(static_cast<float>(v));
            std::memcpy(dst, &h, 2);
        } else if (s.bits == 32) {
            const uint32_t u = static_cast<uint32_t>(v);
            std::memcpy(dst, &u, 4);
        } else if (s.bits == 16) {
            const uint16_t u = static_cast<uint16_t>(v);
            std::memcpy(dst, &u, 2);
        } else {
            *dst = static_cast<uint8_t>(v);
        }
    };
    if (s.tiled) {
        const uint32_t tile = 16;
        TIFFSetField(tif.get(), TIFFTAG_TILEWIDTH, tile);
        TIFFSetField(tif.get(), TIFFTAG_TILELENGTH, tile);
        std::vector<uint8_t> buf(static_cast<size_t>(tile) * tile * s.samples * bytes);
        for (uint32_t ty = 0; ty < s.height; ty += tile)
            for (uint32_t tx = 0; tx < s.width; tx += tile) {
                std::fill(buf.begin(), buf.end(), uint8_t{0});
                for (uint32_t y = 0; y < tile && ty + y < s.height; ++y)
                    for (uint32_t x = 0; x < tile && tx + x < s.width; ++x)
                        for (uint32_t c = 0; c < s.samples; ++c) store(buf.data() + ((static_cast<size_t>(y) * tile + x) * s.samples + c) * bytes, value(tx + x, ty + y, c));
                TIFFWriteTile(tif.get(), buf.data(), tx, ty, 0, 0);
            }
        return;
    }
    TIFFSetField(tif.get(), TIFFTAG_ROWSPERSTRIP, s.height);
    if (s.planarSeparate) {
        std::vector<uint8_t> row(static_cast<size_t>(s.width) * bytes);
        for (uint32_t c = 0; c < s.samples; ++c)
            for (uint32_t y = 0; y < s.height; ++y) {
                for (uint32_t x = 0; x < s.width; ++x) store(row.data() + static_cast<size_t>(x) * bytes, value(x, y, c));
                TIFFWriteScanline(tif.get(), row.data(), y, static_cast<uint16_t>(c));
            }
        return;
    }
    std::vector<uint8_t> row(static_cast<size_t>(s.width) * s.samples * bytes);
    for (uint32_t y = 0; y < s.height; ++y) {
        for (uint32_t x = 0; x < s.width; ++x)
            for (uint32_t c = 0; c < s.samples; ++c) store(row.data() + (static_cast<size_t>(x) * s.samples + c) * bytes, value(x, y, c));
        TIFFWriteScanline(tif.get(), row.data(), y, 0);
    }
}

// An EXR of HALF channels with a display window displayW x displayH at (0, 0) and a data window of dataW x dataH at (dataX, dataY).
inline void WriteExrWindows(const std::filesystem::path& path, uint32_t displayW, uint32_t displayH, int dataX, int dataY, uint32_t dataW, uint32_t dataH,
                            const std::vector<std::string>& channels, const std::function<float(int, int, size_t)>& value) {
    const Imath::Box2i display(Imath::V2i(0, 0), Imath::V2i(static_cast<int>(displayW) - 1, static_cast<int>(displayH) - 1));
    const Imath::Box2i data(Imath::V2i(dataX, dataY), Imath::V2i(dataX + static_cast<int>(dataW) - 1, dataY + static_cast<int>(dataH) - 1));
    Imf::Header header(display, data);
    for (const auto& c : channels) header.channels().insert(c, Imf::Channel(Imf::HALF));
    std::vector<std::vector<uint16_t>> planes(channels.size(), std::vector<uint16_t>(static_cast<size_t>(dataW) * dataH));
    Imf::FrameBuffer fb;
    for (size_t c = 0; c < channels.size(); ++c) {
        for (uint32_t y = 0; y < dataH; ++y)
            for (uint32_t x = 0; x < dataW; ++x) planes[c][static_cast<size_t>(y) * dataW + x] = FloatToHalf(value(dataX + static_cast<int>(x), dataY + static_cast<int>(y), c));
        char* base = reinterpret_cast<char*>(planes[c].data()) - (static_cast<ptrdiff_t>(dataY) * dataW + dataX) * 2;
        fb.insert(channels[c], Imf::Slice(Imf::HALF, base, 2, static_cast<size_t>(dataW) * 2));
    }
    Imf::OutputFile file(ExrFileName(path).c_str(), header);  // UTF-8, as OpenEXR reads a char* name on Windows
    file.setFrameBuffer(fb);
    file.writePixels(static_cast<int>(dataH));
}

// A PNG of only a signature, an IHDR of w x h (8-bit grey) and an empty IDAT: the header reads, the pixels do not.
inline void WritePngHeaderOnly(const std::filesystem::path& path, uint32_t w, uint32_t h) {
    std::ofstream out(path, std::ios::binary);
    const unsigned char sig[8] = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1A, '\n'};
    out.write(reinterpret_cast<const char*>(sig), 8);
    auto be32 = [](uint32_t v, unsigned char* p) {
        p[0] = static_cast<unsigned char>(v >> 24);
        p[1] = static_cast<unsigned char>(v >> 16);
        p[2] = static_cast<unsigned char>(v >> 8);
        p[3] = static_cast<unsigned char>(v);
    };
    auto chunk = [&](const char* type, const std::vector<unsigned char>& body) {
        unsigned char len[4], crcBytes[4];
        be32(static_cast<uint32_t>(body.size()), len);
        out.write(reinterpret_cast<const char*>(len), 4);
        std::vector<unsigned char> typed(type, type + 4);
        typed.insert(typed.end(), body.begin(), body.end());
        out.write(reinterpret_cast<const char*>(typed.data()), static_cast<std::streamsize>(typed.size()));
        be32(static_cast<uint32_t>(crc32(0, typed.data(), static_cast<uInt>(typed.size()))), crcBytes);
        out.write(reinterpret_cast<const char*>(crcBytes), 4);
    };
    std::vector<unsigned char> ihdr(13, 0);
    be32(w, ihdr.data());
    be32(h, ihdr.data() + 4);
    ihdr[8] = 8;  // bit depth; colour type 0 (grey), deflate, filter 0, no interlace
    chunk("IHDR", ihdr);
    chunk("IDAT", {});
    chunk("IEND", {});
}

}  // namespace dlssvid::test
