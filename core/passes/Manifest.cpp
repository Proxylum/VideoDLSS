#include "passes/Manifest.h"

#include <cstdio>
#include <fstream>

#include "util/Error.h"

namespace dlssvid {

std::string_view ToString(FileFormat f) {
    switch (f) {
        case FileFormat::Exr: return "exr";
        case FileFormat::Png: return "png16";
        case FileFormat::Tiff: return "tiff";
        case FileFormat::Npz: return "npz";
        case FileFormat::Raw: return "raw";
    }
    return "?";
}

std::optional<FileFormat> ParseFileFormat(std::string_view s) {
    if (s == "exr") return FileFormat::Exr;
    if (s == "png" || s == "png16") return FileFormat::Png;
    if (s == "tif" || s == "tiff") return FileFormat::Tiff;
    if (s == "npz" || s == "npy") return FileFormat::Npz;
    if (s == "raw" || s == "r32" || s == "rg16f" || s == "bin") return FileFormat::Raw;
    return std::nullopt;
}

std::optional<FileFormat> FormatFromExtension(const std::filesystem::path& p) {
    std::string ext = p.extension().string();
    if (!ext.empty() && ext[0] == '.') ext.erase(0, 1);
    for (auto& c : ext) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (ext == "exr") return FileFormat::Exr;
    if (ext == "png") return FileFormat::Png;
    if (ext == "tif" || ext == "tiff") return FileFormat::Tiff;
    if (ext == "npz" || ext == "npy") return FileFormat::Npz;
    if (ext == "r32" || ext == "rg16f" || ext == "r16f" || ext == "rg32f" || ext == "r8" || ext == "bin") return FileFormat::Raw;
    return std::nullopt;
}

std::string ExtensionFor(FileFormat f, PixelType type, size_t channels) {
    switch (f) {
        case FileFormat::Exr: return ".exr";
        case FileFormat::Png: return ".png";
        case FileFormat::Tiff: return ".tif";
        case FileFormat::Npz: return ".npz";
        case FileFormat::Raw: {
            static const char* prefix[] = {"r", "rg", "rgb", "rgba"};
            std::string ext = ".";
            ext += channels >= 1 && channels <= 4 ? prefix[channels - 1] : "c" + std::to_string(channels);
            switch (type) {
                case PixelType::U8: ext += "8"; break;
                case PixelType::U16: ext += "16"; break;
                case PixelType::F16: ext += "16f"; break;
                case PixelType::F32: ext += "32f"; break;
            }
            if (ext == ".r32f") ext = ".r32";  // ТЗ §5 names: .r32 (depth) and .rg16f (mv)
            return ext;
        }
    }
    return ".bin";
}

Manifest Manifest::ForPass(PassKind kind, uint32_t w, uint32_t h, FileFormat format) {
    const PassSpec& s = Spec(kind);
    Manifest m;
    m.pass = s.name;
    m.width = w;
    m.height = h;
    m.format = format;
    m.pixelType = s.type;
    if (format == FileFormat::Png) m.pixelType = s.isColor ? PixelType::U16 : (s.kind == PassKind::Mask ? PixelType::U8 : PixelType::U16);
    if (format == FileFormat::Raw && kind == PassKind::MvDlss) m.pixelType = PixelType::F16;  // RG16F
    m.channels = s.channels;
    m.convention = s.convention;
    m.colorspace = s.isColor ? "srgb" : "";
    m.filePattern = std::string(s.name) + "_%06d" + ExtensionFor(format, m.pixelType, m.channels.size());
    if (kind == PassKind::MvDlss) m.mv.direction = "backward";
    if (kind == PassKind::MvRaw) m.mv.direction = "forward";
    return m;
}

nlohmann::json Manifest::ToJson() const {
    nlohmann::json j;
    j["schema_version"] = schemaVersion;
    j["pass"] = pass;
    j["width"] = width;
    j["height"] = height;
    j["fps"] = {{"num", fps.num}, {"den", fps.den}};
    j["frame_count"] = frameCount;
    j["frame_range"] = {{"first", firstFrame}, {"last", lastFrame}};
    j["format"] = std::string(ToString(format));
    j["file_pattern"] = filePattern;
    j["pixel_type"] = std::string(ToString(pixelType));
    j["channels"] = channels;
    j["colorspace"] = colorspace;
    j["convention"] = std::string(ToString(convention));
    j["model"] = {{"name", model}, {"version", modelVersion}};
    j["source"] = {{"file", sourceFile}, {"hash", sourceHash}};
    j["stage_params"] = stageParams;
    j["depth"] = {{"units", depth.units}, {"relative", depth.relative}, {"near", depth.zNear},
                  {"far", depth.zFar},     {"min", depth.minValue},      {"max", depth.maxValue}};
    j["mv"] = {{"direction", mv.direction}, {"y_up", mv.yUp}, {"ref_width", mv.refWidth}, {"ref_height", mv.refHeight}};
    if (!fingerprint.empty()) {  // only passes stamped by ProcessRunner carry the version block
        j["fingerprint"] = fingerprint;
        j["inputs"] = inputs;
        j["tool"] = tool;
        j["params_canonical"] = paramsCanonical;
        j["created"] = created;
    }
    return j;
}

Manifest Manifest::FromJson(const nlohmann::json& j) {
    Manifest m;
    m.schemaVersion = j.value("schema_version", 0);
    if (m.schemaVersion != kSchemaVersion)
        Throw("manifest schema_version " + std::to_string(m.schemaVersion) + " is not supported (expected " +
              std::to_string(kSchemaVersion) + ")");
    m.pass = j.value("pass", "");
    m.width = j.value("width", 0u);
    m.height = j.value("height", 0u);
    if (j.contains("fps")) m.fps = Rational{j["fps"].value("num", 0), j["fps"].value("den", 1)};
    m.frameCount = j.value("frame_count", int64_t{0});
    if (j.contains("frame_range")) {
        m.firstFrame = j["frame_range"].value("first", int64_t{0});
        m.lastFrame = j["frame_range"].value("last", int64_t{-1});
    }
    const auto fmt = ParseFileFormat(j.value("format", "exr"));
    if (!fmt) Throw("manifest: unknown format " + j.value("format", ""));
    m.format = *fmt;
    m.filePattern = j.value("file_pattern", "");
    const auto pt = ParsePixelType(j.value("pixel_type", "f32"));
    if (!pt) Throw("manifest: unknown pixel_type " + j.value("pixel_type", ""));
    m.pixelType = *pt;
    m.channels = j.value("channels", std::vector<std::string>{});
    m.colorspace = j.value("colorspace", "");
    const auto conv = ParseConvention(j.value("convention", "raw"));
    if (!conv) Throw("manifest: unknown convention " + j.value("convention", ""));
    m.convention = *conv;
    if (j.contains("model")) {
        m.model = j["model"].value("name", "");
        m.modelVersion = j["model"].value("version", "");
    }
    if (j.contains("source")) {
        m.sourceFile = j["source"].value("file", "");
        m.sourceHash = j["source"].value("hash", "");
    }
    m.stageParams = j.value("stage_params", nlohmann::json::object());
    if (j.contains("depth")) {
        const auto& d = j["depth"];
        m.depth.units = d.value("units", "meters");
        m.depth.relative = d.value("relative", false);
        m.depth.zNear = d.value("near", 0.1f);
        m.depth.zFar = d.value("far", 1000.f);
        m.depth.minValue = d.value("min", 0.f);
        m.depth.maxValue = d.value("max", 0.f);
    }
    if (j.contains("mv")) {
        const auto& v = j["mv"];
        m.mv.direction = v.value("direction", "forward");
        m.mv.yUp = v.value("y_up", false);
        m.mv.refWidth = v.value("ref_width", 0u);
        m.mv.refHeight = v.value("ref_height", 0u);
    }
    m.fingerprint = j.value("fingerprint", "");
    if (j.contains("inputs") && j["inputs"].is_object())
        for (const auto& [k, v] : j["inputs"].items()) m.inputs[k] = v.is_string() ? v.get<std::string>() : "";
    m.tool = j.value("tool", nlohmann::json::object());
    m.paramsCanonical = j.value("params_canonical", nlohmann::json::object());
    m.created = j.value("created", "");
    if (m.width == 0 || m.height == 0) Throw("manifest: width/height missing");
    if (m.filePattern.empty()) m.filePattern = m.pass + "_%06d" + ExtensionFor(m.format, m.pixelType, m.channels.size());
    return m;
}

void Manifest::Save(const std::filesystem::path& dir) const {
    std::filesystem::create_directories(dir);
    std::ofstream out(dir / kFileName, std::ios::binary);
    if (!out) Throw("cannot write " + (dir / kFileName).string());
    out << ToJson().dump(2) << "\n";
}

bool Manifest::Exists(const std::filesystem::path& dir) { return std::filesystem::exists(dir / kFileName); }

Manifest Manifest::Load(const std::filesystem::path& dir) {
    std::ifstream in(dir / kFileName, std::ios::binary);
    if (!in) Throw("no manifest.json in " + dir.string());
    nlohmann::json j;
    try {
        in >> j;
    } catch (const std::exception& e) {
        Throw("manifest.json parse error in " + dir.string() + ": " + e.what());
    }
    return FromJson(j);
}

std::string Manifest::FrameFileName(int64_t frame) const {
    char buf[512];
    std::snprintf(buf, sizeof(buf), filePattern.c_str(), static_cast<long long>(frame));
    return buf;
}

}  // namespace dlssvid
