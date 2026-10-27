#include "passes/formats/ImageProbe.h"

#include <OpenEXR/ImfChannelList.h>
#include <OpenEXR/ImfHeader.h>
#include <OpenEXR/ImfInputFile.h>
#include <png.h>
#include <tiffio.h>

#pragma warning(disable : 4611)  // setjmp with C++ objects: standard libpng usage, no C++ objects are destroyed on longjmp

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>

#include "passes/formats/ExrIO.h"
#include "util/Error.h"

namespace dlssvid {

namespace {

// the texts of the probe in UTF-8: path.string() throws for a character outside the ANSI code page
std::string Utf8(const std::filesystem::path& p) { return ExrFileName(p); }

void OnPngError(png_structp png, png_const_charp msg) {
    std::string* err = static_cast<std::string*>(png_get_error_ptr(png));
    if (err) *err = msg ? msg : "png error";
    png_longjmp(png, 1);
}
void OnPngWarning(png_structp, png_const_charp) {}

void ProbePng(const std::filesystem::path& path, ImageProbe& p) {
    FILE* raw = _wfopen(path.wstring().c_str(), L"rb");
    if (!raw) Throw("cannot open " + Utf8(path));
    std::unique_ptr<FILE, int (*)(FILE*)> f(raw, &std::fclose);
    std::string err;
    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, &err, OnPngError, OnPngWarning);
    if (!png) Throw("png_create_read_struct failed");
    png_infop info = png_create_info_struct(png);
    if (!info) {
        png_destroy_read_struct(&png, nullptr, nullptr);
        Throw("png_create_info_struct failed");
    }
    if (setjmp(png_jmpbuf(png))) {
        png_destroy_read_struct(&png, &info, nullptr);
        Throw("PNG header does not read (" + Utf8(path) + "): " + err);
    }
    png_init_io(png, f.get());
    png_read_info(png, info);
    p.width = png_get_image_width(png, info);
    p.height = png_get_image_height(png, info);
    const int depth = png_get_bit_depth(png, info), color = png_get_color_type(png, info);
    const bool trns = png_get_valid(png, info, PNG_INFO_tRNS) != 0;
    png_destroy_read_struct(&png, &info, nullptr);
    // what ReadPng returns: the palette expanded to RGB, tRNS to an alpha channel, gray below 8 bits to 8
    switch (color) {
        case PNG_COLOR_TYPE_GRAY: p.channels = trns ? 2 : 1; break;
        case PNG_COLOR_TYPE_GRAY_ALPHA: p.channels = 2; break;
        case PNG_COLOR_TYPE_PALETTE:
        case PNG_COLOR_TYPE_RGB: p.channels = trns ? 4 : 3; break;
        default: p.channels = 4; break;
    }
    p.type = depth == 16 ? PixelType::U16 : PixelType::U8;
}

void ProbeTiff(const std::filesystem::path& path, ImageProbe& p) {
    TIFFSetWarningHandler(nullptr);
    std::unique_ptr<TIFF, void (*)(TIFF*)> tif(TIFFOpenW(path.wstring().c_str(), "r"), &TIFFClose);
    if (!tif) Throw("TIFF header does not read: " + Utf8(path));
    uint32_t w = 0, h = 0;
    uint16_t bits = 0, spp = 1, fmt = SAMPLEFORMAT_UINT, planar = PLANARCONFIG_CONTIG, photometric = 0;
    TIFFGetField(tif.get(), TIFFTAG_IMAGEWIDTH, &w);
    TIFFGetField(tif.get(), TIFFTAG_IMAGELENGTH, &h);
    TIFFGetFieldDefaulted(tif.get(), TIFFTAG_BITSPERSAMPLE, &bits);
    TIFFGetFieldDefaulted(tif.get(), TIFFTAG_SAMPLESPERPIXEL, &spp);
    TIFFGetFieldDefaulted(tif.get(), TIFFTAG_SAMPLEFORMAT, &fmt);
    TIFFGetFieldDefaulted(tif.get(), TIFFTAG_PLANARCONFIG, &planar);
    if (!TIFFGetField(tif.get(), TIFFTAG_PHOTOMETRIC, &photometric)) photometric = spp >= 3 ? PHOTOMETRIC_RGB : PHOTOMETRIC_MINISBLACK;
    p.width = w;
    p.height = h;
    p.channels = spp;
    p.tiffPhotometric = photometric;
    p.tiffTiled = TIFFIsTiled(tif.get()) != 0;
    p.tiffPlanarSeparate = planar == PLANARCONFIG_SEPARATE;
    if (fmt == SAMPLEFORMAT_IEEEFP && bits == 32) p.type = PixelType::F32;
    else if (fmt == SAMPLEFORMAT_IEEEFP && bits == 16) p.type = PixelType::F16;
    else if (fmt == SAMPLEFORMAT_UINT && bits == 16) p.type = PixelType::U16;
    else if (fmt == SAMPLEFORMAT_UINT && bits == 8) p.type = PixelType::U8;
    else p.tiffSamplesSupported = false;
}

void ProbeExr(const std::filesystem::path& path, ImageProbe& p) {
    try {
        Imf::InputFile file(ExrFileName(path).c_str());  // UTF-8: a mask from a Cyrillic folder
        const Imf::Header& header = file.header();
        const Imath::Box2i display = header.displayWindow(), data = header.dataWindow();
        p.width = static_cast<uint32_t>(std::max(0, display.max.x - display.min.x + 1));
        p.height = static_cast<uint32_t>(std::max(0, display.max.y - display.min.y + 1));
        p.exrDataMinX = data.min.x - display.min.x;
        p.exrDataMinY = data.min.y - display.min.y;
        p.exrDataWidth = static_cast<uint32_t>(std::max(0, data.max.x - data.min.x + 1));
        p.exrDataHeight = static_cast<uint32_t>(std::max(0, data.max.y - data.min.y + 1));
        bool wide = false;
        uint32_t top = 0;
        for (auto it = header.channels().begin(); it != header.channels().end(); ++it) {
            const std::string name = it.name();
            p.exrChannels.push_back(name);
            if (name.find('.') == std::string::npos) ++top;
            if (it.channel().type != Imf::HALF) wide = true;
        }
        p.channels = top ? top : static_cast<uint32_t>(p.exrChannels.size());
        p.type = wide ? PixelType::F32 : PixelType::F16;
    } catch (const std::exception& e) {
        Throw("EXR header does not read (" + ExrFileName(path) + "): " + e.what());
    }
}

}  // namespace

ImageProbe::Format ImageFormatOf(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) Throw("cannot open " + Utf8(path));  // access denied, held without share-read: unreadable, not another kind
    unsigned char s[4] = {};
    if (!in.read(reinterpret_cast<char*>(s), 4)) return ImageProbe::Format::Unknown;
    if (s[0] == 0x89 && s[1] == 'P' && s[2] == 'N' && s[3] == 'G') return ImageProbe::Format::Png;
    if ((s[0] == 'I' && s[1] == 'I' && s[2] == '*' && s[3] == 0) || (s[0] == 'M' && s[1] == 'M' && s[2] == 0 && s[3] == '*')) return ImageProbe::Format::Tiff;
    if (s[0] == 0x76 && s[1] == 0x2F && s[2] == 0x31 && s[3] == 0x01) return ImageProbe::Format::Exr;
    return ImageProbe::Format::Unknown;
}

std::string_view ToString(ImageProbe::Format f) {
    switch (f) {
        case ImageProbe::Format::Png: return "png";
        case ImageProbe::Format::Tiff: return "tiff";
        case ImageProbe::Format::Exr: return "exr";
        default: return "unknown";
    }
}

ImageProbe ProbeImage(const std::filesystem::path& path) {
    ImageProbe p;
    p.format = ImageFormatOf(path);
    switch (p.format) {
        case ImageProbe::Format::Png: ProbePng(path, p); break;
        case ImageProbe::Format::Tiff: ProbeTiff(path, p); break;
        case ImageProbe::Format::Exr: ProbeExr(path, p); break;
        default: break;
    }
    return p;
}

}  // namespace dlssvid
