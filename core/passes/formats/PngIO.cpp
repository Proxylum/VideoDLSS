#include "passes/formats/PngIO.h"

#include <png.h>

#pragma warning(disable : 4611)  // setjmp with C++ objects: standard libpng usage, no C++ objects are destroyed on longjmp

#include <cstdio>
#include <vector>

#include "util/Error.h"

namespace dlssvid {

namespace {

struct FileCloser {
    void operator()(FILE* f) const {
        if (f) std::fclose(f);
    }
};
using FilePtr = std::unique_ptr<FILE, FileCloser>;

FilePtr OpenFile(const std::filesystem::path& path, const wchar_t* mode) {
    FILE* f = _wfopen(path.wstring().c_str(), mode);
    if (!f) Throw("cannot open " + path.string());
    return FilePtr(f);
}

void OnPngError(png_structp png, png_const_charp msg) {
    std::string* err = static_cast<std::string*>(png_get_error_ptr(png));
    if (err) *err = msg ? msg : "png error";
    png_longjmp(png, 1);
}
void OnPngWarning(png_structp, png_const_charp) {}

std::vector<std::string> ChannelsFor(int n) {
    switch (n) {
        case 1: return {"A"};
        case 2: return {"Y", "A"};
        case 3: return {"R", "G", "B"};
        default: return {"R", "G", "B", "A"};
    }
}

}  // namespace

PassImage ReadPng(const std::filesystem::path& path) {
    FilePtr f = OpenFile(path, L"rb");
    std::string err;
    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, &err, OnPngError, OnPngWarning);
    if (!png) Throw("png_create_read_struct failed");
    png_infop info = png_create_info_struct(png);
    if (!info) {
        png_destroy_read_struct(&png, nullptr, nullptr);
        Throw("png_create_info_struct failed");
    }
    PassImage img;
    std::vector<png_bytep> rows;
    if (setjmp(png_jmpbuf(png))) {
        png_destroy_read_struct(&png, &info, nullptr);
        Throw("PNG read failed (" + path.string() + "): " + err);
    }
    png_init_io(png, f.get());
    png_read_info(png, info);

    const png_uint_32 w = png_get_image_width(png, info), h = png_get_image_height(png, info);
    const int depth = png_get_bit_depth(png, info);
    const int color = png_get_color_type(png, info);
    if (color == PNG_COLOR_TYPE_PALETTE) png_set_palette_to_rgb(png);
    if (color == PNG_COLOR_TYPE_GRAY && depth < 8) png_set_expand_gray_1_2_4_to_8(png);
    if (png_get_valid(png, info, PNG_INFO_tRNS)) png_set_tRNS_to_alpha(png);
    if (depth == 16) png_set_swap(png);  // PNG is big-endian; we want host (little) order
    png_read_update_info(png, info);

    const int channels = png_get_channels(png, info);
    const int outDepth = png_get_bit_depth(png, info);
    img.Allocate(w, h, outDepth == 16 ? PixelType::U16 : PixelType::U8, ChannelsFor(channels));
    if (png_get_rowbytes(png, info) != img.RowBytes()) {
        png_destroy_read_struct(&png, &info, nullptr);
        Throw("PNG row size mismatch: " + path.string());
    }
    rows.resize(h);
    for (png_uint_32 y = 0; y < h; ++y) rows[y] = img.data.data() + static_cast<size_t>(y) * img.RowBytes();
    png_read_image(png, rows.data());
    png_read_end(png, nullptr);
    png_destroy_read_struct(&png, &info, nullptr);
    return img;
}

void WritePng(const std::filesystem::path& path, const PassImage& img) {
    if (img.type != PixelType::U8 && img.type != PixelType::U16) Throw("WritePng: only u8/u16 images (convert first)");
    if (img.channels.empty() || img.channels.size() > 4) Throw("WritePng: 1-4 channels supported");
    static const int colorTypes[] = {PNG_COLOR_TYPE_GRAY, PNG_COLOR_TYPE_GRAY_ALPHA, PNG_COLOR_TYPE_RGB, PNG_COLOR_TYPE_RGB_ALPHA};

    FilePtr f = OpenFile(path, L"wb");
    std::string err;
    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, &err, OnPngError, OnPngWarning);
    if (!png) Throw("png_create_write_struct failed");
    png_infop info = png_create_info_struct(png);
    if (!info) {
        png_destroy_write_struct(&png, nullptr);
        Throw("png_create_info_struct failed");
    }
    std::vector<png_bytep> rows(img.height);
    if (setjmp(png_jmpbuf(png))) {
        png_destroy_write_struct(&png, &info);
        Throw("PNG write failed (" + path.string() + "): " + err);
    }
    png_init_io(png, f.get());
    png_set_IHDR(png, info, img.width, img.height, img.type == PixelType::U16 ? 16 : 8, colorTypes[img.channels.size() - 1],
                 PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_compression_level(png, 6);
    png_write_info(png, info);
    if (img.type == PixelType::U16) png_set_swap(png);
    for (uint32_t y = 0; y < img.height; ++y)
        rows[y] = const_cast<png_bytep>(img.data.data() + static_cast<size_t>(y) * img.RowBytes());
    png_write_image(png, rows.data());
    png_write_end(png, nullptr);
    png_destroy_write_struct(&png, &info);
}

}  // namespace dlssvid
