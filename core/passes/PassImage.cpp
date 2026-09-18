#include "passes/PassImage.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "util/Error.h"
#include "util/Half.h"

namespace dlssvid {

size_t BytesPer(PixelType t) {
    switch (t) {
        case PixelType::U8: return 1;
        case PixelType::U16: return 2;
        case PixelType::F16: return 2;
        case PixelType::F32: return 4;
    }
    return 0;
}

std::string_view ToString(PixelType t) {
    switch (t) {
        case PixelType::U8: return "u8";
        case PixelType::U16: return "u16";
        case PixelType::F16: return "f16";
        case PixelType::F32: return "f32";
    }
    return "?";
}

std::optional<PixelType> ParsePixelType(std::string_view s) {
    if (s == "u8") return PixelType::U8;
    if (s == "u16") return PixelType::U16;
    if (s == "f16" || s == "half") return PixelType::F16;
    if (s == "f32" || s == "float") return PixelType::F32;
    return std::nullopt;
}

void PassImage::Allocate(uint32_t w, uint32_t h, PixelType t, std::vector<std::string> ch) {
    if (w == 0 || h == 0) Throw("PassImage::Allocate: zero-sized image");
    if (ch.empty()) Throw("PassImage::Allocate: no channels");
    width = w;
    height = h;
    type = t;
    channels = std::move(ch);
    data.assign(ByteSize(), 0);
}

int PassImage::ChannelIndex(std::string_view name) const {
    for (size_t i = 0; i < channels.size(); ++i)
        if (channels[i] == name) return static_cast<int>(i);
    return -1;
}

float PassImage::Get(uint32_t x, uint32_t y, size_t c) const {
    const size_t i = (static_cast<size_t>(y) * width + x) * channels.size() + c;
    switch (type) {
        case PixelType::U8: return static_cast<float>(As<uint8_t>()[i]);
        case PixelType::U16: return static_cast<float>(As<uint16_t>()[i]);
        case PixelType::F16: return HalfToFloat(As<uint16_t>()[i]);
        case PixelType::F32: return As<float>()[i];
    }
    return 0.f;
}

void PassImage::Set(uint32_t x, uint32_t y, size_t c, float v) {
    const size_t i = (static_cast<size_t>(y) * width + x) * channels.size() + c;
    switch (type) {
        case PixelType::U8: As<uint8_t>()[i] = static_cast<uint8_t>(std::clamp(std::lround(v), 0L, 255L)); break;
        case PixelType::U16: As<uint16_t>()[i] = static_cast<uint16_t>(std::clamp(std::lround(v), 0L, 65535L)); break;
        case PixelType::F16: As<uint16_t>()[i] = FloatToHalf(v); break;
        case PixelType::F32: As<float>()[i] = v; break;
    }
}

PassImage PassImage::ConvertTo(PixelType t) const {
    if (t == type) return *this;
    PassImage out;
    out.Allocate(width, height, t, channels);
    const size_t n = static_cast<size_t>(width) * height * channels.size();
    for (size_t i = 0; i < n; ++i) {
        float v = 0.f;
        switch (type) {
            case PixelType::U8: v = static_cast<float>(As<uint8_t>()[i]); break;
            case PixelType::U16: v = static_cast<float>(As<uint16_t>()[i]); break;
            case PixelType::F16: v = HalfToFloat(As<uint16_t>()[i]); break;
            case PixelType::F32: v = As<float>()[i]; break;
        }
        switch (t) {
            case PixelType::U8: out.As<uint8_t>()[i] = static_cast<uint8_t>(std::clamp(std::lround(v), 0L, 255L)); break;
            case PixelType::U16: out.As<uint16_t>()[i] = static_cast<uint16_t>(std::clamp(std::lround(v), 0L, 65535L)); break;
            case PixelType::F16: out.As<uint16_t>()[i] = FloatToHalf(v); break;
            case PixelType::F32: out.As<float>()[i] = v; break;
        }
    }
    return out;
}

double PassImage::MaxAbsDiff(const PassImage& a, const PassImage& b) {
    if (!a.SameLayout(b)) Throw("PassImage::MaxAbsDiff: layout mismatch");
    double m = 0.0;
    for (uint32_t y = 0; y < a.height; ++y)
        for (uint32_t x = 0; x < a.width; ++x)
            for (size_t c = 0; c < a.channels.size(); ++c) {
                const double d = std::fabs(static_cast<double>(a.Get(x, y, c)) - b.Get(x, y, c));
                if (d > m || std::isnan(d)) m = std::isnan(d) ? INFINITY : d;
            }
    return m;
}

// ---- pass kinds ----------------------------------------------------------------------

namespace {
const std::vector<PassSpec>& Specs() {
    static const std::vector<PassSpec> specs = {
        {PassKind::ColorSource, "color_source", {"R", "G", "B"}, PixelType::F16, Convention::Raw, true},
        {PassKind::DepthRaw, "depth_raw", {"Z"}, PixelType::F32, Convention::Raw, false},
        {PassKind::DepthDlss, "depth_dlss", {"Z"}, PixelType::F32, Convention::Dlss, false},
        {PassKind::MvRaw, "mv_raw", {"u", "v"}, PixelType::F32, Convention::Raw, false},
        {PassKind::MvDlss, "mv_dlss", {"u", "v"}, PixelType::F32, Convention::Dlss, false},
        {PassKind::Mask, "mask", {"A"}, PixelType::U8, Convention::Raw, false},
        {PassKind::ColorSr, "color_sr", {"R", "G", "B"}, PixelType::F16, Convention::Raw, true},
        {PassKind::ColorNr, "color_nr", {"R", "G", "B"}, PixelType::F16, Convention::Raw, true},
        {PassKind::ColorFg, "color_fg", {"R", "G", "B"}, PixelType::F16, Convention::Raw, true},
        {PassKind::Result, "result", {"R", "G", "B"}, PixelType::F16, Convention::Raw, true},
    };
    return specs;
}
}  // namespace

const PassSpec& Spec(PassKind kind) {
    for (const auto& s : Specs())
        if (s.kind == kind) return s;
    Throw("unknown PassKind");
}

std::string_view ToString(PassKind kind) { return Spec(kind).name; }

std::optional<PassKind> ParsePassKind(std::string_view name) {
    for (const auto& s : Specs())
        if (name == s.name) return s.kind;
    return std::nullopt;
}

std::string_view ToString(Convention c) { return c == Convention::Raw ? "raw" : "dlss"; }

std::optional<Convention> ParseConvention(std::string_view s) {
    if (s == "raw") return Convention::Raw;
    if (s == "dlss") return Convention::Dlss;
    return std::nullopt;
}

const std::vector<PassKind>& AllPassKinds() {
    static const std::vector<PassKind> kinds = [] {
        std::vector<PassKind> k;
        for (const auto& s : Specs()) k.push_back(s.kind);
        return k;
    }();
    return kinds;
}

PassImage MakePassImage(PassKind kind, uint32_t w, uint32_t h, std::optional<PixelType> type) {
    const PassSpec& s = Spec(kind);
    PassImage img;
    img.Allocate(w, h, type.value_or(s.type), s.channels);
    return img;
}

}  // namespace dlssvid
