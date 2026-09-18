#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace dlssvid {

// ---- pixel storage --------------------------------------------------------------------

enum class PixelType : uint8_t { U8, U16, F16, F32 };

size_t BytesPer(PixelType t);
std::string_view ToString(PixelType t);
std::optional<PixelType> ParsePixelType(std::string_view s);

// CPU image of one pass for one frame: interleaved channels, row-major, tightly packed.
struct PassImage {
    uint32_t width = 0;
    uint32_t height = 0;
    PixelType type = PixelType::F32;
    std::vector<std::string> channels;  // e.g. {"R","G","B"}, {"Z"}, {"u","v"}, {"A"}
    std::vector<uint8_t> data;

    void Allocate(uint32_t w, uint32_t h, PixelType t, std::vector<std::string> ch);

    size_t ChannelCount() const { return channels.size(); }
    size_t PixelBytes() const { return BytesPer(type) * channels.size(); }
    size_t RowBytes() const { return PixelBytes() * width; }
    size_t ByteSize() const { return RowBytes() * height; }
    bool Empty() const { return data.empty(); }
    bool SameLayout(const PassImage& o) const {
        return width == o.width && height == o.height && type == o.type && channels == o.channels;
    }
    int ChannelIndex(std::string_view name) const;

    template <class T>
    T* As() { return reinterpret_cast<T*>(data.data()); }
    template <class T>
    const T* As() const { return reinterpret_cast<const T*>(data.data()); }

    // Generic element access with conversion (U8/U16 as raw integer values, F16 via half).
    float Get(uint32_t x, uint32_t y, size_t c) const;
    void Set(uint32_t x, uint32_t y, size_t c, float v);

    // Deep conversion of the sample type; integer <-> float conversions are value-preserving
    // (no normalisation), float -> integer rounds and clamps.
    PassImage ConvertTo(PixelType t) const;

    // Max absolute difference over all samples (both images must have the same layout).
    static double MaxAbsDiff(const PassImage& a, const PassImage& b);
};

// ---- pass kinds ----------------------------------------------------------------------

enum class PassKind : uint8_t {
    ColorSource,
    DepthRaw,
    DepthDlss,
    MvRaw,
    MvDlss,
    Mask,
    ColorSr,
    ColorNr,
    ColorFg,
    Result,
};

enum class Convention : uint8_t { Raw, Dlss };

struct PassSpec {
    PassKind kind;
    const char* name;                   // canonical name used in manifests and file names
    std::vector<std::string> channels;  // canonical channel names
    PixelType type;                     // canonical storage type
    Convention convention;
    bool isColor;
};

const PassSpec& Spec(PassKind kind);
std::string_view ToString(PassKind kind);
std::optional<PassKind> ParsePassKind(std::string_view name);
std::string_view ToString(Convention c);
std::optional<Convention> ParseConvention(std::string_view s);
const std::vector<PassKind>& AllPassKinds();

// Allocates an image with the canonical layout of a pass (optionally overriding the sample type).
PassImage MakePassImage(PassKind kind, uint32_t w, uint32_t h, std::optional<PixelType> type = std::nullopt);

}  // namespace dlssvid
