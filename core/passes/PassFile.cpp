#include "passes/PassFile.h"

#include "passes/formats/ExrIO.h"
#include "passes/formats/NpzIO.h"
#include "passes/formats/PngIO.h"
#include "passes/formats/RawIO.h"
#include "passes/formats/TiffIO.h"
#include "util/Error.h"

namespace dlssvid {

namespace {

// Rename generic channels ("A", "Y"...) to the manifest's canonical names when counts match.
void ApplyManifestChannels(PassImage& img, const Manifest* m) {
    if (m && !m->channels.empty() && m->channels.size() == img.channels.size()) img.channels = m->channels;
}

}  // namespace

PassImage ReadPassFile(const std::filesystem::path& path, const Manifest* manifest) {
    const auto fmt = FormatFromExtension(path);
    if (!fmt) Throw("unknown pass file extension: " + path.string());
    PassImage img;
    switch (*fmt) {
        case FileFormat::Exr: img = ReadExr(path); break;
        case FileFormat::Png: img = ReadPng(path); break;
        case FileFormat::Tiff: img = ReadTiff(path); break;
        case FileFormat::Npz: {
            auto arrays = ReadNpz(path);
            if (arrays.empty()) Throw("npz has no arrays: " + path.string());
            auto it = manifest ? arrays.find(manifest->pass) : arrays.end();
            img = it != arrays.end() ? std::move(it->second) : std::move(arrays.begin()->second);
            break;
        }
        case FileFormat::Raw: {
            if (!manifest) Throw("raw pass file needs a manifest for geometry: " + path.string());
            img = ReadRaw(path, manifest->width, manifest->height, manifest->pixelType, manifest->channels);
            break;
        }
    }
    ApplyManifestChannels(img, manifest);
    // Restore the canonical sample type when the container widened it (EXR stores u8 as half, u16 as float).
    if (manifest && manifest->pixelType != img.type && *fmt == FileFormat::Exr) img = img.ConvertTo(manifest->pixelType);
    return img;
}

void WritePassFile(const std::filesystem::path& path, const PassImage& img) {
    const auto fmt = FormatFromExtension(path);
    if (!fmt) Throw("unknown pass file extension: " + path.string());
    switch (*fmt) {
        case FileFormat::Exr: WriteExr(path, img); break;
        case FileFormat::Png: WritePng(path, img); break;
        case FileFormat::Tiff: WriteTiff(path, img); break;
        case FileFormat::Npz: {
            const std::string key = path.stem().string();
            // strip the "_NNNNNN" frame suffix so the array key is the pass name
            const size_t us = key.rfind('_');
            WriteNpz(path, {{us == std::string::npos ? key : key.substr(0, us), img}});
            break;
        }
        case FileFormat::Raw: WriteRaw(path, img); break;
    }
}

PassImage PrepareForFormat(const PassImage& img, FileFormat format, PixelType wanted) {
    if (format == FileFormat::Png && wanted != PixelType::U8 && wanted != PixelType::U16)
        Throw("PNG cannot store " + std::string(ToString(wanted)) + " samples");
    return img.type == wanted ? img : img.ConvertTo(wanted);
}

}  // namespace dlssvid
