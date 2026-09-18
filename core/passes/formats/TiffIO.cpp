#include "passes/formats/TiffIO.h"

#include <tiffio.h>

#include <cstring>
#include <memory>
#include <vector>

#include "util/Error.h"

namespace dlssvid {

namespace {

struct TiffCloser {
    void operator()(TIFF* t) const {
        if (t) TIFFClose(t);
    }
};
using TiffPtr = std::unique_ptr<TIFF, TiffCloser>;

std::vector<std::string> ChannelsFor(int n) {
    switch (n) {
        case 1: return {"A"};
        case 2: return {"Y", "A"};
        case 3: return {"R", "G", "B"};
        default: return {"R", "G", "B", "A"};
    }
}

}  // namespace

PassImage ReadTiff(const std::filesystem::path& path) {
    TIFFSetWarningHandler(nullptr);
    TiffPtr tif(TIFFOpenW(path.wstring().c_str(), "r"));
    if (!tif) Throw("cannot open TIFF " + path.string());
    uint32_t w = 0, h = 0;
    uint16_t bits = 0, spp = 1, fmt = SAMPLEFORMAT_UINT, planar = PLANARCONFIG_CONTIG;
    TIFFGetField(tif.get(), TIFFTAG_IMAGEWIDTH, &w);
    TIFFGetField(tif.get(), TIFFTAG_IMAGELENGTH, &h);
    TIFFGetField(tif.get(), TIFFTAG_BITSPERSAMPLE, &bits);
    TIFFGetFieldDefaulted(tif.get(), TIFFTAG_SAMPLESPERPIXEL, &spp);
    TIFFGetFieldDefaulted(tif.get(), TIFFTAG_SAMPLEFORMAT, &fmt);
    TIFFGetFieldDefaulted(tif.get(), TIFFTAG_PLANARCONFIG, &planar);
    if (planar != PLANARCONFIG_CONTIG) Throw("TIFF: only contiguous planar config supported: " + path.string());
    PixelType type;
    if (fmt == SAMPLEFORMAT_IEEEFP && bits == 32) type = PixelType::F32;
    else if (fmt == SAMPLEFORMAT_IEEEFP && bits == 16) type = PixelType::F16;
    else if (fmt == SAMPLEFORMAT_UINT && bits == 16) type = PixelType::U16;
    else if (fmt == SAMPLEFORMAT_UINT && bits == 8) type = PixelType::U8;
    else Throw("TIFF: unsupported sample format/bits " + std::to_string(fmt) + "/" + std::to_string(bits) + ": " + path.string());
    if (spp < 1 || spp > 4) Throw("TIFF: 1-4 samples per pixel supported: " + path.string());

    PassImage img;
    img.Allocate(w, h, type, ChannelsFor(spp));
    const tmsize_t scan = TIFFScanlineSize(tif.get());
    if (static_cast<size_t>(scan) != img.RowBytes()) Throw("TIFF scanline size mismatch: " + path.string());
    for (uint32_t y = 0; y < h; ++y) {
        if (TIFFReadScanline(tif.get(), img.data.data() + static_cast<size_t>(y) * img.RowBytes(), y, 0) < 0)
            Throw("TIFF read failed at row " + std::to_string(y) + ": " + path.string());
    }
    return img;
}

void WriteTiff(const std::filesystem::path& path, const PassImage& img) {
    if (img.channels.empty() || img.channels.size() > 4) Throw("WriteTiff: 1-4 channels supported");
    TIFFSetWarningHandler(nullptr);
    TiffPtr tif(TIFFOpenW(path.wstring().c_str(), "w"));
    if (!tif) Throw("cannot create TIFF " + path.string());
    const uint16_t spp = static_cast<uint16_t>(img.channels.size());
    const uint16_t bits = static_cast<uint16_t>(BytesPer(img.type) * 8);
    const uint16_t fmt = (img.type == PixelType::F16 || img.type == PixelType::F32) ? SAMPLEFORMAT_IEEEFP : SAMPLEFORMAT_UINT;
    TIFFSetField(tif.get(), TIFFTAG_IMAGEWIDTH, img.width);
    TIFFSetField(tif.get(), TIFFTAG_IMAGELENGTH, img.height);
    TIFFSetField(tif.get(), TIFFTAG_BITSPERSAMPLE, bits);
    TIFFSetField(tif.get(), TIFFTAG_SAMPLESPERPIXEL, spp);
    TIFFSetField(tif.get(), TIFFTAG_SAMPLEFORMAT, fmt);
    TIFFSetField(tif.get(), TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
    TIFFSetField(tif.get(), TIFFTAG_PHOTOMETRIC, spp >= 3 ? PHOTOMETRIC_RGB : PHOTOMETRIC_MINISBLACK);
    if (spp == 2 || spp == 4) {
        const uint16_t extra[] = {EXTRASAMPLE_UNASSALPHA};
        TIFFSetField(tif.get(), TIFFTAG_EXTRASAMPLES, 1, extra);
    }
    TIFFSetField(tif.get(), TIFFTAG_COMPRESSION, COMPRESSION_ADOBE_DEFLATE);
    TIFFSetField(tif.get(), TIFFTAG_ROWSPERSTRIP, TIFFDefaultStripSize(tif.get(), 0));
    TIFFSetField(tif.get(), TIFFTAG_SOFTWARE, "dlssvid");
    std::vector<uint8_t> row(img.RowBytes());
    for (uint32_t y = 0; y < img.height; ++y) {
        std::memcpy(row.data(), img.data.data() + static_cast<size_t>(y) * img.RowBytes(), row.size());
        if (TIFFWriteScanline(tif.get(), row.data(), y, 0) < 0) Throw("TIFF write failed at row " + std::to_string(y) + ": " + path.string());
    }
}

}  // namespace dlssvid
