#include "passes/PassSequence.h"

#include <algorithm>
#include <charconv>
#include <map>
#include <sstream>

#include "passes/PassFile.h"
#include "passes/formats/ExrIO.h"
#include "util/Error.h"
#include "util/Log.h"

namespace dlssvid {

// ---- PassWriter ------------------------------------------------------------------------

PassWriter::PassWriter(const std::filesystem::path& dir, Manifest manifest) : dir_(dir), manifest_(std::move(manifest)) {
    if (manifest_.width == 0 || manifest_.height == 0) Throw("PassWriter: manifest width/height missing");
    if (manifest_.channels.empty()) Throw("PassWriter: manifest channels missing");
    if (manifest_.filePattern.empty())
        manifest_.filePattern = manifest_.pass + "_%06d" + ExtensionFor(manifest_.format, manifest_.pixelType, manifest_.channels.size());
    if (manifest_.format == FileFormat::Png && manifest_.pixelType != PixelType::U8 && manifest_.pixelType != PixelType::U16)
        Throw("PassWriter: PNG cannot store " + std::string(ToString(manifest_.pixelType)));
    std::filesystem::create_directories(dir_);
    manifest_.frameCount = 0;
    manifest_.firstFrame = 0;
    manifest_.lastFrame = -1;
}

PassWriter::~PassWriter() {
    if (!finished_ && count_ > 0) {
        try {
            Finish();
        } catch (const std::exception& e) {
            Log()->error("PassWriter::Finish in destructor: {}", e.what());
        }
    }
}

void PassWriter::WriteFrame(int64_t frame, const PassImage& img) {
    if (finished_) Throw("PassWriter: WriteFrame after Finish");
    if (img.width != manifest_.width || img.height != manifest_.height)
        Throw("PassWriter: frame " + std::to_string(frame) + " is " + std::to_string(img.width) + "x" + std::to_string(img.height) +
              ", pass is " + std::to_string(manifest_.width) + "x" + std::to_string(manifest_.height));
    if (img.channels.size() != manifest_.channels.size())
        Throw("PassWriter: frame " + std::to_string(frame) + " has " + std::to_string(img.channels.size()) + " channels, pass has " +
              std::to_string(manifest_.channels.size()));
    PassImage stored = PrepareForFormat(img, manifest_.format, manifest_.pixelType);
    stored.channels = manifest_.channels;
    WritePassFile(dir_ / manifest_.FrameFileName(frame), stored);
    if (count_ == 0) {
        manifest_.firstFrame = manifest_.lastFrame = frame;
    } else {
        manifest_.firstFrame = std::min(manifest_.firstFrame, frame);
        manifest_.lastFrame = std::max(manifest_.lastFrame, frame);
    }
    ++count_;
}

void PassWriter::Finish() {
    if (finished_) return;
    manifest_.frameCount = count_;
    manifest_.Save(dir_);
    finished_ = true;
}

// ---- PassReader ------------------------------------------------------------------------

PassReader PassReader::Open(const std::filesystem::path& dir) {
    if (!std::filesystem::is_directory(dir)) Throw("pass folder not found: " + dir.string());
    PassReader r;
    r.dir_ = dir;
    r.manifest_ = Manifest::Load(dir);
    r.hadManifest_ = true;
    return r;
}

PassReader PassReader::OpenWithoutManifest(const std::filesystem::path& dir, const ScanHints& hints) {
    if (!std::filesystem::is_directory(dir)) Throw("pass folder not found: " + dir.string());
    // Collect numbered files of one supported format.
    std::map<int64_t, std::filesystem::path> frames;
    std::optional<FileFormat> format;
    std::string prefix, ext;
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        if (!entry.is_regular_file()) continue;
        const auto fmt = FormatFromExtension(entry.path());
        if (!fmt) continue;
        const std::string stem = entry.path().stem().string();
        size_t digits = 0;
        while (digits < stem.size() && std::isdigit(static_cast<unsigned char>(stem[stem.size() - 1 - digits]))) ++digits;
        if (digits == 0) continue;
        if (format && *fmt != *format) continue;
        int64_t n = 0;
        std::from_chars(stem.data() + stem.size() - digits, stem.data() + stem.size(), n);
        if (!format) {
            format = fmt;
            prefix = stem.substr(0, stem.size() - digits);
            ext = entry.path().extension().string();
        }
        if (stem.compare(0, prefix.size(), prefix) != 0) continue;
        frames[n] = entry.path();
    }
    if (frames.empty()) Throw("no numbered pass files found in " + dir.string());

    const PassSpec& spec = Spec(hints.kind);
    Manifest m;
    m.pass = spec.name;
    m.format = *format;
    m.convention = hints.convention;
    m.channels = spec.channels;
    m.pixelType = hints.pixelType.value_or(spec.type);
    m.filePattern = prefix + "%0" + std::to_string(frames.begin()->second.stem().string().size() - prefix.size()) + "lld" + ext;
    m.firstFrame = frames.begin()->first;
    m.lastFrame = frames.rbegin()->first;
    m.frameCount = static_cast<int64_t>(frames.size());
    m.colorspace = spec.isColor ? "srgb" : "";
    if (hints.kind == PassKind::MvDlss) m.mv.direction = "backward";

    if (*format == FileFormat::Raw) {
        if (hints.width == 0 || hints.height == 0) Throw("raw pass files need --size WxH when there is no manifest: " + dir.string());
        m.width = hints.width;
        m.height = hints.height;
    } else {
        const PassImage first = ReadPassFile(frames.begin()->second, nullptr);
        m.width = first.width;
        m.height = first.height;
        if (first.channels.size() != spec.channels.size())
            Throw("pass " + std::string(spec.name) + " expects " + std::to_string(spec.channels.size()) + " channels, file has " +
                  std::to_string(first.channels.size()) + ": " + frames.begin()->second.string());
        m.pixelType = first.type == PixelType::F16 && spec.type == PixelType::F32 ? PixelType::F16 : m.pixelType;
        if (first.type == PixelType::U8 || first.type == PixelType::U16) m.pixelType = first.type;
    }
    // Sanity: the pattern must reproduce the first file name.
    if (m.FrameFileName(m.firstFrame) != frames.begin()->second.filename().string()) {
        // fall back to a lenient pattern using the detected digit width
        m.filePattern = prefix + "%lld" + ext;
    }
    PassReader r;
    r.dir_ = dir;
    r.manifest_ = m;
    r.hadManifest_ = false;
    return r;
}

PassImage PassReader::ReadFrame(int64_t frame) const {
    const auto path = FramePath(frame);
    if (!std::filesystem::exists(path)) Throw("missing pass frame " + std::to_string(frame) + ": " + path.string());
    PassImage img = ReadPassFile(path, &manifest_);
    if (img.width != manifest_.width || img.height != manifest_.height)
        Throw("pass frame " + std::to_string(frame) + " is " + std::to_string(img.width) + "x" + std::to_string(img.height) +
              " but the manifest says " + std::to_string(manifest_.width) + "x" + std::to_string(manifest_.height));
    return img;
}

std::vector<int64_t> PassReader::MissingFrames() const {
    std::vector<int64_t> missing;
    for (int64_t f = manifest_.firstFrame; f <= manifest_.lastFrame; ++f)
        if (!HasFrame(f)) missing.push_back(f);
    return missing;
}

void PassReader::Validate(const Expect& expect) const {
    std::vector<std::string> problems;
    if (expect.width && expect.height && (manifest_.width != expect.width || manifest_.height != expect.height))
        problems.push_back("resolution " + std::to_string(manifest_.width) + "x" + std::to_string(manifest_.height) + " (expected " +
                           std::to_string(expect.width) + "x" + std::to_string(expect.height) + ")");
    if (expect.frameCount >= 0 && manifest_.frameCount != expect.frameCount)
        problems.push_back("frame count " + std::to_string(manifest_.frameCount) + " (expected " + std::to_string(expect.frameCount) + ")");
    if (expect.kind && manifest_.pass != Spec(*expect.kind).name)
        problems.push_back("pass '" + manifest_.pass + "' (expected '" + std::string(Spec(*expect.kind).name) + "')");
    const auto missing = MissingFrames();
    if (!missing.empty()) {
        std::string list;
        for (size_t i = 0; i < missing.size() && i < 10; ++i) list += (i ? ", " : "") + std::to_string(missing[i]);
        if (missing.size() > 10) list += ", ... (" + std::to_string(missing.size()) + " total)";
        problems.push_back("missing frames: " + list);
    }
    if (!problems.empty()) {
        std::string msg = "pass '" + manifest_.pass + "' in " + dir_.string() + " does not match:";
        for (const auto& p : problems) msg += "\n  - " + p;
        Throw(msg);
    }
}

// ---- export ---------------------------------------------------------------------------

std::optional<FrameRange> FrameRange::Parse(std::string_view s) {
    FrameRange r;
    const size_t dash = s.find('-');
    auto toInt = [](std::string_view t, int64_t& out) {
        return !t.empty() && std::from_chars(t.data(), t.data() + t.size(), out).ec == std::errc{};
    };
    if (dash == std::string_view::npos) {
        if (!toInt(s, r.first)) return std::nullopt;
        r.last = r.first;
        return r;
    }
    if (!toInt(s.substr(0, dash), r.first)) return std::nullopt;
    const auto rest = s.substr(dash + 1);
    if (rest.empty()) {
        r.last = -1;
        return r;
    }
    if (!toInt(rest, r.last) || r.last < r.first) return std::nullopt;
    return r;
}

int64_t ExportPass(const std::filesystem::path& dir, Manifest manifest, const FrameSource& source, FrameRange range, const ProgressFn& progress) {
    PassWriter writer(dir, std::move(manifest));
    const int64_t total = range.last >= 0 ? range.last - range.first + 1 : -1;
    PassImage img;
    int64_t done = 0;
    for (int64_t f = range.first; range.last < 0 || f <= range.last; ++f) {
        if (!source(f, img)) {
            if (range.last >= 0) Throw("export: source has no frame " + std::to_string(f));
            break;
        }
        writer.WriteFrame(f, img);
        ++done;
        if (progress) progress(done, total);
    }
    writer.Finish();
    return done;
}

FrameSource SourceFromReader(const PassReader& reader) {
    return [&reader](int64_t frame, PassImage& out) {
        if (!reader.HasFrame(frame)) return false;
        out = reader.ReadFrame(frame);
        return true;
    };
}

int64_t ExportLayeredExr(const std::filesystem::path& dir, const std::string& name, Manifest base, const FrameSource& color,
                         const FrameSource* depth, const FrameSource* mv, FrameRange range, const ProgressFn& progress) {
    std::filesystem::create_directories(dir);
    base.pass = name;
    base.format = FileFormat::Exr;
    base.pixelType = PixelType::F16;
    base.channels = {"R", "G", "B"};
    if (depth) base.channels.push_back("depth.Z");
    if (mv) {
        base.channels.push_back("mv.u");
        base.channels.push_back("mv.v");
    }
    base.filePattern = name + "_%06d.exr";
    base.colorspace = base.colorspace.empty() ? "srgb" : base.colorspace;
    base.frameCount = 0;
    base.firstFrame = 0;
    base.lastFrame = -1;

    const int64_t total = range.last >= 0 ? range.last - range.first + 1 : -1;
    int64_t done = 0;
    PassImage c, d, m;
    for (int64_t f = range.first; range.last < 0 || f <= range.last; ++f) {
        if (!color(f, c)) {
            if (range.last >= 0) Throw("export: colour source has no frame " + std::to_string(f));
            break;
        }
        if (c.width != base.width || c.height != base.height) Throw("export: colour frame size mismatch at " + std::to_string(f));
        ExrLayers layers;
        layers.emplace_back("", c.type == PixelType::F16 || c.type == PixelType::F32 ? c : c.ConvertTo(PixelType::F16));
        if (depth) {
            if (!(*depth)(f, d)) Throw("export: depth source has no frame " + std::to_string(f));
            layers.emplace_back("depth", d);
        }
        if (mv) {
            if (!(*mv)(f, m)) Throw("export: mv source has no frame " + std::to_string(f));
            layers.emplace_back("mv", m);
        }
        WriteExrLayers(dir / base.FrameFileName(f), layers);
        if (done == 0) base.firstFrame = f;
        base.lastFrame = f;
        ++done;
        if (progress) progress(done, total);
    }
    base.frameCount = done;
    base.Save(dir);
    return done;
}

std::optional<ExportPreset> ParseExportPreset(std::string_view s) {
    if (s.empty() || s == "none") return ExportPreset::None;
    if (s == "nuke" || s == "resolve") return ExportPreset::Nuke;
    if (s == "comfyui" || s == "comfy") return ExportPreset::ComfyUi;
    if (s == "rawdlss" || s == "raw-dlss" || s == "dlss") return ExportPreset::RawDlss;
    return std::nullopt;
}

FileFormat PresetFormat(ExportPreset preset, PassKind kind) {
    const bool color = Spec(kind).isColor;
    switch (preset) {
        case ExportPreset::None:
        case ExportPreset::Nuke: return FileFormat::Exr;
        case ExportPreset::ComfyUi: return color || kind == PassKind::Mask ? FileFormat::Png : FileFormat::Npz;
        case ExportPreset::RawDlss: return FileFormat::Raw;
    }
    return FileFormat::Exr;
}

PixelType PresetPixelType(ExportPreset preset, PassKind kind) {
    const PassSpec& s = Spec(kind);
    switch (preset) {
        case ExportPreset::None:
        case ExportPreset::Nuke: return s.type;
        case ExportPreset::ComfyUi: return s.isColor ? PixelType::U16 : (kind == PassKind::Mask ? PixelType::U8 : PixelType::F32);
        case ExportPreset::RawDlss: return kind == PassKind::MvDlss ? PixelType::F16 : s.type;
    }
    return s.type;
}

}  // namespace dlssvid
