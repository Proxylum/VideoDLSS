#include "passes/formats/ExrIO.h"

#include <OpenEXR/ImfChannelList.h>
#include <OpenEXR/ImfFrameBuffer.h>
#include <OpenEXR/ImfHeader.h>
#include <OpenEXR/ImfInputFile.h>
#include <OpenEXR/ImfOutputFile.h>
#include <OpenEXR/ImfStringAttribute.h>

#include <algorithm>

#include "util/Error.h"

namespace dlssvid {

namespace {

Imf::PixelType ExrTypeFor(PixelType t) {
    switch (t) {
        case PixelType::U8:
        case PixelType::F16: return Imf::HALF;
        case PixelType::U16:
        case PixelType::F32: return Imf::FLOAT;
    }
    return Imf::FLOAT;
}

// The image as it is laid out on disk (U8 -> F16, U16 -> F32).
PassImage Storable(const PassImage& img) {
    if (img.type == PixelType::U8) return img.ConvertTo(PixelType::F16);
    if (img.type == PixelType::U16) return img.ConvertTo(PixelType::F32);
    return img;
}

std::string ChannelName(const std::string& layer, const std::string& ch) { return layer.empty() ? ch : layer + "." + ch; }

// Canonical channel order inside a layer so that R,G,B and u,v come back in the expected order.
int ChannelRank(const std::string& c) {
    static const char* order[] = {"R", "G", "B", "A", "Z", "u", "v", "X", "Y"};
    for (int i = 0; i < static_cast<int>(std::size(order)); ++i)
        if (c == order[i]) return i;
    return 100;
}

}  // namespace

void WriteExrLayers(const std::filesystem::path& path, const ExrLayers& layers) {
    if (layers.empty()) Throw("WriteExrLayers: no layers");
    const uint32_t w = layers.front().second.width, h = layers.front().second.height;
    std::vector<PassImage> stored;
    stored.reserve(layers.size());
    for (const auto& [layer, img] : layers) {
        if (img.width != w || img.height != h) Throw("WriteExrLayers: layer size mismatch in " + layer);
        stored.push_back(Storable(img));
    }

    Imf::Header header(static_cast<int>(w), static_cast<int>(h));
    header.compression() = Imf::ZIP_COMPRESSION;
    header.insert("dlssvid", Imf::StringAttribute("pass-sequence"));
    Imf::FrameBuffer fb;
    for (size_t i = 0; i < layers.size(); ++i) {
        const PassImage& img = stored[i];
        const Imf::PixelType et = ExrTypeFor(img.type);
        for (size_t c = 0; c < img.channels.size(); ++c) {
            const std::string name = ChannelName(layers[i].first, img.channels[c]);
            header.channels().insert(name, Imf::Channel(et));
            fb.insert(name, Imf::Slice(et, const_cast<char*>(reinterpret_cast<const char*>(img.data.data() + c * BytesPer(img.type))),
                                       img.PixelBytes(), img.RowBytes()));
        }
    }
    try {
        Imf::OutputFile file(path.string().c_str(), header);
        file.setFrameBuffer(fb);
        file.writePixels(static_cast<int>(h));
    } catch (const std::exception& e) {
        Throw("EXR write failed (" + path.string() + "): " + e.what());
    }
}

void WriteExr(const std::filesystem::path& path, const PassImage& img) { WriteExrLayers(path, {{"", img}}); }

std::map<std::string, PassImage> ReadExrLayers(const std::filesystem::path& path) {
    std::map<std::string, PassImage> out;
    try {
        Imf::InputFile file(path.string().c_str());
        const Imf::Header& header = file.header();
        const Imath::Box2i dw = header.dataWindow();
        const uint32_t w = static_cast<uint32_t>(dw.max.x - dw.min.x + 1);
        const uint32_t h = static_cast<uint32_t>(dw.max.y - dw.min.y + 1);

        // Group channels by layer and pick a common sample type per layer (FLOAT if mixed).
        struct LayerInfo {
            std::vector<std::string> channels;
            bool anyFloat = false;
            bool anyUint = false;
        };
        std::map<std::string, LayerInfo> layers;
        for (auto it = header.channels().begin(); it != header.channels().end(); ++it) {
            const std::string full = it.name();
            const size_t dot = full.rfind('.');
            const std::string layer = dot == std::string::npos ? "" : full.substr(0, dot);
            const std::string ch = dot == std::string::npos ? full : full.substr(dot + 1);
            LayerInfo& li = layers[layer];
            li.channels.push_back(ch);
            if (it.channel().type == Imf::FLOAT) li.anyFloat = true;
            if (it.channel().type == Imf::UINT) li.anyUint = true;
        }

        Imf::FrameBuffer fb;
        for (auto& [layer, li] : layers) {
            std::sort(li.channels.begin(), li.channels.end(),
                      [](const std::string& a, const std::string& b) { return ChannelRank(a) != ChannelRank(b) ? ChannelRank(a) < ChannelRank(b) : a < b; });
            PassImage img;
            const PixelType t = (li.anyFloat || li.anyUint) ? PixelType::F32 : PixelType::F16;
            img.Allocate(w, h, t, li.channels);
            const Imf::PixelType et = t == PixelType::F32 ? Imf::FLOAT : Imf::HALF;
            out[layer] = std::move(img);
            PassImage& ref = out[layer];
            for (size_t c = 0; c < ref.channels.size(); ++c) {
                char* base = reinterpret_cast<char*>(ref.data.data() + c * BytesPer(ref.type));
                // Slice base must account for the data window origin.
                base -= (static_cast<ptrdiff_t>(dw.min.y) * static_cast<ptrdiff_t>(ref.RowBytes()) + static_cast<ptrdiff_t>(dw.min.x) * static_cast<ptrdiff_t>(ref.PixelBytes()));
                fb.insert(ChannelName(layer, ref.channels[c]), Imf::Slice(et, base, ref.PixelBytes(), ref.RowBytes()));
            }
        }
        file.setFrameBuffer(fb);
        file.readPixels(dw.min.y, dw.max.y);
    } catch (const Error&) {
        throw;
    } catch (const std::exception& e) {
        Throw("EXR read failed (" + path.string() + "): " + e.what());
    }
    return out;
}

PassImage ReadExr(const std::filesystem::path& path) {
    auto layers = ReadExrLayers(path);
    if (layers.empty()) Throw("EXR has no channels: " + path.string());
    if (layers.size() == 1) return std::move(layers.begin()->second);
    // Single-pass read of a multi-layer file: prefer the top-level layer.
    auto it = layers.find("");
    if (it != layers.end()) return std::move(it->second);
    return std::move(layers.begin()->second);
}

}  // namespace dlssvid
