#include "passes/formats/RawIO.h"

#include <fstream>

#include "util/Error.h"

namespace dlssvid {

PassImage ReadRaw(const std::filesystem::path& path, uint32_t width, uint32_t height, PixelType type,
                  const std::vector<std::string>& channels) {
    PassImage img;
    img.Allocate(width, height, type, channels);
    std::ifstream in(path, std::ios::binary);
    if (!in) Throw("cannot open " + path.string());
    in.seekg(0, std::ios::end);
    const auto size = static_cast<size_t>(in.tellg());
    if (size != img.ByteSize())
        Throw("raw pass size mismatch for " + path.string() + ": file has " + std::to_string(size) + " bytes, expected " +
              std::to_string(img.ByteSize()) + " (" + std::to_string(width) + "x" + std::to_string(height) + " " +
              std::string(ToString(type)) + " x" + std::to_string(channels.size()) + ")");
    in.seekg(0);
    in.read(reinterpret_cast<char*>(img.data.data()), static_cast<std::streamsize>(img.ByteSize()));
    return img;
}

void WriteRaw(const std::filesystem::path& path, const PassImage& img) {
    std::ofstream out(path, std::ios::binary);
    if (!out) Throw("cannot write " + path.string());
    out.write(reinterpret_cast<const char*>(img.data.data()), static_cast<std::streamsize>(img.ByteSize()));
}

}  // namespace dlssvid
