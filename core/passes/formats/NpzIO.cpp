#include "passes/formats/NpzIO.h"

#include <zlib.h>

#include <cstring>
#include <fstream>
#include <regex>
#include <vector>

#include "util/Error.h"

namespace dlssvid {

namespace {

std::vector<uint8_t> ReadFile(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) Throw("cannot open " + path.string());
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

void WriteFile(const std::filesystem::path& path, const std::vector<uint8_t>& bytes) {
    std::ofstream out(path, std::ios::binary);
    if (!out) Throw("cannot write " + path.string());
    out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
}

const char* Descr(PixelType t) {
    switch (t) {
        case PixelType::U8: return "|u1";
        case PixelType::U16: return "<u2";
        case PixelType::F16: return "<f2";
        case PixelType::F32: return "<f4";
    }
    return "<f4";
}

std::optional<PixelType> TypeFromDescr(const std::string& d) {
    if (d == "|u1" || d == "u1" || d == "<u1") return PixelType::U8;
    if (d == "<u2") return PixelType::U16;
    if (d == "<f2") return PixelType::F16;
    if (d == "<f4") return PixelType::F32;
    return std::nullopt;
}

std::vector<std::string> ChannelsFor(size_t n) {
    switch (n) {
        case 1: return {"A"};
        case 2: return {"u", "v"};
        case 3: return {"R", "G", "B"};
        case 4: return {"R", "G", "B", "A"};
        default: {
            std::vector<std::string> c;
            for (size_t i = 0; i < n; ++i) c.push_back("c" + std::to_string(i));
            return c;
        }
    }
}

void Put16(std::vector<uint8_t>& v, uint16_t x) {
    v.push_back(static_cast<uint8_t>(x & 0xFF));
    v.push_back(static_cast<uint8_t>(x >> 8));
}
void Put32(std::vector<uint8_t>& v, uint32_t x) {
    Put16(v, static_cast<uint16_t>(x & 0xFFFF));
    Put16(v, static_cast<uint16_t>(x >> 16));
}
uint16_t Get16(const uint8_t* p) { return static_cast<uint16_t>(p[0] | (p[1] << 8)); }
uint32_t Get32(const uint8_t* p) { return static_cast<uint32_t>(p[0] | (p[1] << 8) | (p[2] << 16) | (static_cast<uint32_t>(p[3]) << 24)); }

std::vector<uint8_t> Inflate(const uint8_t* src, size_t srcLen, size_t dstLen) {
    std::vector<uint8_t> out(dstLen);
    z_stream zs{};
    if (inflateInit2(&zs, -MAX_WBITS) != Z_OK) Throw("inflateInit2 failed");
    zs.next_in = const_cast<Bytef*>(src);
    zs.avail_in = static_cast<uInt>(srcLen);
    zs.next_out = out.data();
    zs.avail_out = static_cast<uInt>(dstLen);
    const int rc = inflate(&zs, Z_FINISH);
    inflateEnd(&zs);
    if (rc != Z_STREAM_END) Throw("npz: deflate stream is corrupt");
    return out;
}

}  // namespace

std::vector<uint8_t> EncodeNpy(const PassImage& img) {
    std::string shape = "(" + std::to_string(img.height) + ", " + std::to_string(img.width);
    if (img.channels.size() > 1) shape += ", " + std::to_string(img.channels.size());
    shape += img.channels.size() > 1 ? ")" : ")";
    if (img.channels.size() == 1) shape = "(" + std::to_string(img.height) + ", " + std::to_string(img.width) + ")";
    std::string dict = "{'descr': '" + std::string(Descr(img.type)) + "', 'fortran_order': False, 'shape': " + shape + ", }";
    // Total header (magic 6 + version 2 + len 2 + dict + '\n') must be a multiple of 64.
    size_t total = 10 + dict.size() + 1;
    const size_t pad = (64 - total % 64) % 64;
    dict.append(pad, ' ');
    dict.push_back('\n');
    std::vector<uint8_t> out;
    out.reserve(10 + dict.size() + img.ByteSize());
    const uint8_t magic[] = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0};
    out.insert(out.end(), magic, magic + 8);
    Put16(out, static_cast<uint16_t>(dict.size()));
    out.insert(out.end(), dict.begin(), dict.end());
    out.insert(out.end(), img.data.begin(), img.data.end());
    return out;
}

PassImage DecodeNpy(const uint8_t* data, size_t size, const std::string& what) {
    if (size < 10 || std::memcmp(data, "\x93NUMPY", 6) != 0) Throw("not a .npy array: " + what);
    const int major = data[6];
    size_t headerLen, off;
    if (major == 1) {
        headerLen = Get16(data + 8);
        off = 10;
    } else {
        headerLen = Get32(data + 8);
        off = 12;
    }
    if (off + headerLen > size) Throw("npy header truncated: " + what);
    const std::string header(reinterpret_cast<const char*>(data + off), headerLen);
    std::smatch m;
    if (!std::regex_search(header, m, std::regex("'descr'\\s*:\\s*'([^']+)'"))) Throw("npy: no descr: " + what);
    const auto type = TypeFromDescr(m[1]);
    if (!type) Throw("npy: unsupported dtype " + m[1].str() + ": " + what);
    if (std::regex_search(header, m, std::regex("'fortran_order'\\s*:\\s*True"))) Throw("npy: fortran order not supported: " + what);
    if (!std::regex_search(header, m, std::regex("'shape'\\s*:\\s*\\(([^)]*)\\)"))) Throw("npy: no shape: " + what);
    std::vector<size_t> shape;
    {
        const std::string s = m[1];
        std::regex num("\\d+");
        for (auto it = std::sregex_iterator(s.begin(), s.end(), num); it != std::sregex_iterator(); ++it) shape.push_back(std::stoull(it->str()));
    }
    if (shape.size() < 2 || shape.size() > 3) Throw("npy: expected (H, W) or (H, W, C) array: " + what);
    const size_t h = shape[0], w = shape[1], c = shape.size() == 3 ? shape[2] : 1;
    PassImage img;
    img.Allocate(static_cast<uint32_t>(w), static_cast<uint32_t>(h), *type, ChannelsFor(c));
    const size_t dataOff = off + headerLen;
    if (dataOff + img.ByteSize() > size) Throw("npy data truncated: " + what);
    std::memcpy(img.data.data(), data + dataOff, img.ByteSize());
    return img;
}

PassImage ReadNpy(const std::filesystem::path& path) {
    const auto bytes = ReadFile(path);
    return DecodeNpy(bytes.data(), bytes.size(), path.string());
}

void WriteNpy(const std::filesystem::path& path, const PassImage& img) { WriteFile(path, EncodeNpy(img)); }

void WriteNpz(const std::filesystem::path& path, const std::map<std::string, PassImage>& arrays) {
    if (arrays.empty()) Throw("WriteNpz: no arrays");
    std::vector<uint8_t> out, central;
    uint16_t entries = 0;
    for (const auto& [key, img] : arrays) {
        const std::vector<uint8_t> payload = EncodeNpy(img);
        const std::string name = key + ".npy";
        const uint32_t crc = static_cast<uint32_t>(crc32(0L, payload.data(), static_cast<uInt>(payload.size())));
        const uint32_t offset = static_cast<uint32_t>(out.size());
        // local file header
        Put32(out, 0x04034b50);
        Put16(out, 20);  // version needed
        Put16(out, 0);   // flags
        Put16(out, 0);   // stored
        Put16(out, 0);   // time
        Put16(out, 0x21);  // date (1980-01-01)
        Put32(out, crc);
        Put32(out, static_cast<uint32_t>(payload.size()));
        Put32(out, static_cast<uint32_t>(payload.size()));
        Put16(out, static_cast<uint16_t>(name.size()));
        Put16(out, 0);
        out.insert(out.end(), name.begin(), name.end());
        out.insert(out.end(), payload.begin(), payload.end());
        // central directory entry
        Put32(central, 0x02014b50);
        Put16(central, 20);
        Put16(central, 20);
        Put16(central, 0);
        Put16(central, 0);
        Put16(central, 0);
        Put16(central, 0x21);
        Put32(central, crc);
        Put32(central, static_cast<uint32_t>(payload.size()));
        Put32(central, static_cast<uint32_t>(payload.size()));
        Put16(central, static_cast<uint16_t>(name.size()));
        Put16(central, 0);
        Put16(central, 0);
        Put16(central, 0);
        Put16(central, 0);
        Put32(central, 0);
        Put32(central, offset);
        central.insert(central.end(), name.begin(), name.end());
        ++entries;
    }
    const uint32_t cdOffset = static_cast<uint32_t>(out.size());
    out.insert(out.end(), central.begin(), central.end());
    Put32(out, 0x06054b50);
    Put16(out, 0);
    Put16(out, 0);
    Put16(out, entries);
    Put16(out, entries);
    Put32(out, static_cast<uint32_t>(central.size()));
    Put32(out, cdOffset);
    Put16(out, 0);
    WriteFile(path, out);
}

std::map<std::string, PassImage> ReadNpz(const std::filesystem::path& path) {
    const auto bytes = ReadFile(path);
    const size_t n = bytes.size();
    if (n >= 6 && std::memcmp(bytes.data(), "\x93NUMPY", 6) == 0) {  // a bare .npy
        return {{"arr_0", DecodeNpy(bytes.data(), n, path.string())}};
    }
    // Find end of central directory (no comment expected, but scan back a little).
    size_t eocd = std::string::npos;
    for (size_t i = n >= 22 ? n - 22 : 0; i + 4 <= n; --i) {
        if (Get32(bytes.data() + i) == 0x06054b50) {
            eocd = i;
            break;
        }
        if (i == 0) break;
    }
    if (eocd == std::string::npos) Throw("npz: not a zip file: " + path.string());
    const uint16_t entries = Get16(bytes.data() + eocd + 10);
    size_t pos = Get32(bytes.data() + eocd + 16);
    std::map<std::string, PassImage> out;
    for (uint16_t e = 0; e < entries; ++e) {
        if (pos + 46 > n || Get32(bytes.data() + pos) != 0x02014b50) Throw("npz: bad central directory: " + path.string());
        const uint16_t method = Get16(bytes.data() + pos + 10);
        const uint32_t compSize = Get32(bytes.data() + pos + 20);
        const uint32_t uncompSize = Get32(bytes.data() + pos + 24);
        const uint16_t nameLen = Get16(bytes.data() + pos + 28);
        const uint16_t extraLen = Get16(bytes.data() + pos + 30);
        const uint16_t commentLen = Get16(bytes.data() + pos + 32);
        const uint32_t localOff = Get32(bytes.data() + pos + 42);
        std::string name(reinterpret_cast<const char*>(bytes.data() + pos + 46), nameLen);
        pos += 46 + nameLen + extraLen + commentLen;

        if (localOff + 30 > n || Get32(bytes.data() + localOff) != 0x04034b50) Throw("npz: bad local header: " + path.string());
        const uint16_t lNameLen = Get16(bytes.data() + localOff + 26);
        const uint16_t lExtraLen = Get16(bytes.data() + localOff + 28);
        const size_t dataOff = localOff + 30 + lNameLen + lExtraLen;
        if (dataOff + compSize > n) Throw("npz: entry truncated: " + name);
        std::vector<uint8_t> payload;
        if (method == 0) payload.assign(bytes.begin() + static_cast<ptrdiff_t>(dataOff), bytes.begin() + static_cast<ptrdiff_t>(dataOff + compSize));
        else if (method == 8) payload = Inflate(bytes.data() + dataOff, compSize, uncompSize);
        else Throw("npz: unsupported compression method " + std::to_string(method) + " in " + name);
        if (name.size() > 4 && name.compare(name.size() - 4, 4, ".npy") == 0) name.resize(name.size() - 4);
        out[name] = DecodeNpy(payload.data(), payload.size(), path.string() + ":" + name);
    }
    return out;
}

}  // namespace dlssvid
