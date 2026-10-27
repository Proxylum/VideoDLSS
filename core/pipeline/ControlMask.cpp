#include "pipeline/ControlMask.h"

#include <windows.h>  // ERROR_ACCESS_DENIED / ERROR_SHARING_VIOLATION / ERROR_LOCK_VIOLATION
#include <process.h>  // _getpid

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <nlohmann/json.hpp>

#include "passes/Manifest.h"
#include "passes/formats/ExrIO.h"
#include "passes/formats/ImageProbe.h"
#include "passes/formats/PngIO.h"
#include "passes/formats/TiffIO.h"
#include "pipeline/PassFingerprint.h"
#include "pipeline/PassVersions.h"
#include "util/Error.h"
#include "util/Log.h"
#include "util/Sha256.h"

namespace dlssvid {

ControlMaskError::ControlMaskError(ControlMaskIssue i) : std::runtime_error(i.detail), issue(std::move(i)) {}

ControlMaskTestHooks& ControlMaskHooksForTests() {
    static ControlMaskTestHooks hooks;
    return hooks;
}

namespace {

namespace fs = std::filesystem;
using Kind = ControlMaskIssue::Kind;

// TIFF photometric (TIFFTAG_PHOTOMETRIC) the import reads: the value as is, inverted, or R, G, B
constexpr uint16_t kMinIsWhite = 0, kMinIsBlack = 1, kRgb = 2;
constexpr auto kStaleAfter = std::chrono::minutes(10);  // a lock or a working folder older than this is a crashed import's
constexpr const char* kImportDir = ".import";           // no manifest, no ".v": no pass, no history, not seen by gc or the viewport

[[noreturn]] void Fail(Kind k, std::string detail, uint32_t w = 0, uint32_t h = 0) { throw ControlMaskError(ControlMaskIssue{k, std::move(detail), w, h, true}); }

std::string Utf8(const fs::path& p) {
    const std::u8string u = p.u8string();
    return std::string(u.begin(), u.end());
}

// Locked: another program holds a file of the folder (a viewer, an editor) - it can be closed and the import repeated.
bool IsLocked(const std::error_code& ec) {
    return ec.category() == std::system_category() &&
           (ec.value() == ERROR_ACCESS_DENIED || ec.value() == ERROR_SHARING_VIOLATION || ec.value() == ERROR_LOCK_VIOLATION);
}
[[noreturn]] void FailFs(const std::error_code& ec, const std::string& what, bool previousKept = true) {
    throw ControlMaskError(ControlMaskIssue{IsLocked(ec) ? Kind::Locked : Kind::Write, what + ": " + ec.message(), 0, 0, previousKept});
}

bool OlderThan(const fs::path& p, std::chrono::minutes age) {
    std::error_code ec;
    const auto t = fs::last_write_time(p, ec);
    return !ec && fs::file_time_type::clock::now() - t > age;
}

bool ManifestLoads(const fs::path& dir) {
    if (!Manifest::Exists(dir)) return false;
    try {
        Manifest::Load(dir);
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

// The import lock of a passes root: create_directory is atomic, so of two imports into one root (the CLI with the GUI open, two
// windows) the second gets Busy. Released on scope exit, on an exception too.
class ImportLock {
public:
    explicit ImportLock(const fs::path& root) : dir_(root / kImportDir / "lock") {
        std::error_code ec;
        fs::create_directories(root / kImportDir, ec);
        if (ec) FailFs(ec, "cannot create " + Utf8(root / kImportDir));
        for (int attempt = 0; attempt < 2; ++attempt) {
            if (fs::create_directory(dir_, ec)) return;
            if (ec) FailFs(ec, "cannot take the import lock " + Utf8(dir_));
            if (!OlderThan(dir_, kStaleAfter)) break;
            fs::remove_all(dir_, ec);  // a crashed process's
        }
        Fail(Kind::Busy, "another control mask import is running in " + Utf8(root) + ": wait for it and repeat");
    }
    ~ImportLock() {
        std::error_code ec;
        fs::remove_all(dir_, ec);
    }
    ImportLock(const ImportLock&) = delete;
    ImportLock& operator=(const ImportLock&) = delete;

private:
    fs::path dir_;
};

// The working folders of earlier imports that crashed: older than 10 minutes (a fresh one may be another import's, under its lock).
void CleanLeftovers(const fs::path& root) {
    std::error_code ec;
    std::vector<fs::path> stale;
    for (const auto& e : fs::directory_iterator(root / kImportDir, ec))
        if (e.path().filename() != "lock" && OlderThan(e.path(), kStaleAfter)) stale.push_back(e.path());
    for (const auto& p : stale) {
        std::error_code rec;
        fs::remove_all(p, rec);
        if (!rec) Log()->info("mask_control: removed the leftover {}", Utf8(p));
    }
}

struct Decoded {
    PassImage img;  // U16, {"Y"} or {"R", "G", "B"}
    ImageProbe probe;
    bool grayscale = false, bandsDiffer = false, alphaIgnored = false, exrPadded = false, floatSource = false;
    int64_t clamped = 0;
};

void CheckSize(uint32_t w, uint32_t h) {
    if (w == 0 || h == 0) Fail(Kind::Empty, "the image has no pixels");
    if (w > kControlMaskMaxSide || h > kControlMaskMaxSide || static_cast<uint64_t>(w) * h > kControlMaskMaxPixels)
        Fail(Kind::TooLarge, "the image is " + std::to_string(w) + "x" + std::to_string(h) + ": a side may be at most 16384 and the whole at most 2^26 pixels", w, h);
}

// An EXR whose data window is not its display window (a matte from compositing, a bbox): the decoded window on a zero canvas of the
// display window, at its place; what lies outside the display window is cut (outside the data window an EXR is 0 anyway).
PassImage PadToDisplay(const PassImage& data, const ImageProbe& p) {
    PassImage canvas;
    canvas.Allocate(p.width, p.height, data.type, data.channels);
    std::fill(canvas.data.begin(), canvas.data.end(), uint8_t{0});
    const size_t px = data.PixelBytes();
    for (uint32_t y = 0; y < data.height; ++y) {
        const int64_t ty = static_cast<int64_t>(y) + p.exrDataMinY;
        if (ty < 0 || ty >= p.height) continue;
        for (uint32_t x = 0; x < data.width; ++x) {
            const int64_t tx = static_cast<int64_t>(x) + p.exrDataMinX;
            if (tx < 0 || tx >= p.width) continue;
            std::memcpy(canvas.data.data() + (static_cast<size_t>(ty) * p.width + static_cast<size_t>(tx)) * px, data.data.data() + (static_cast<size_t>(y) * data.width + x) * px, px);
        }
    }
    return canvas;
}

// The rules of docs/conventions.md §6: the kind by the signature, the size from the header before decoding, R G B or one grey value
// (alpha dropped), [0, 1] as U16 (U8 x 257 exactly, floats clamped), grey when R = G = B exactly in the source's codes.
Decoded DecodeMask(const fs::path& image) {
    Decoded d;
    try {
        d.probe = ProbeImage(image);
    } catch (const std::exception& e) {
        Fail(Kind::Unreadable, e.what());
    }
    const ImageProbe& p = d.probe;
    if (p.format == ImageProbe::Format::Unknown) Fail(Kind::Unsupported, Utf8(image.filename()) + " is not a PNG, TIFF or EXR image");
    CheckSize(p.width, p.height);
    if (p.format == ImageProbe::Format::Exr) CheckSize(p.exrDataWidth, p.exrDataHeight);
    if (p.format == ImageProbe::Format::Tiff &&
        (p.tiffTiled || p.tiffPlanarSeparate || !p.tiffSamplesSupported ||
         (p.tiffPhotometric != kMinIsWhite && p.tiffPhotometric != kMinIsBlack && p.tiffPhotometric != kRgb)))
        Fail(Kind::Tiff, "this TIFF is not supported (photometric " + std::to_string(p.tiffPhotometric) + (p.tiffTiled ? ", tiles" : "") +
                             (p.tiffPlanarSeparate ? ", separate planes" : "") +
                             (p.tiffSamplesSupported ? "" : ", samples other than 8 / 16-bit unsigned or 16 / 32-bit float (1-bit, 32-bit integer)") +
                             "): save the mask as PNG");
    PassImage src;
    try {
        src = p.format == ImageProbe::Format::Png ? ReadPng(image) : p.format == ImageProbe::Format::Tiff ? ReadTiff(image) : ReadExr(image);
    } catch (const std::exception& e) {
        Fail(Kind::Unreadable, e.what());
    }
    if (p.format == ImageProbe::Format::Exr && (p.exrDataMinX != 0 || p.exrDataMinY != 0 || src.width != p.width || src.height != p.height)) {
        src = PadToDisplay(src, p);
        d.exrPadded = true;
    }

    // which samples: R, G, B or one value; the alpha to drop
    std::vector<size_t> pick;
    int alpha = -1;
    const size_t nc = src.channels.size();
    if (p.format == ImageProbe::Format::Exr) {
        const int r = src.ChannelIndex("R"), g = src.ChannelIndex("G"), b = src.ChannelIndex("B"), y = src.ChannelIndex("Y");
        alpha = src.ChannelIndex("A");
        if (r >= 0 && g >= 0 && b >= 0) pick = {static_cast<size_t>(r), static_cast<size_t>(g), static_cast<size_t>(b)};
        else if (y >= 0) pick = {static_cast<size_t>(y)};
        else if (nc == 1) {
            pick = {0};
            alpha = -1;  // the only channel is the value, whatever its name
        } else {
            std::string names;
            for (const auto& c : src.channels) names += (names.empty() ? "" : ", ") + c;
            Fail(Kind::Channels, "the EXR has no R, G, B, Y or a single channel (" + names + ")");
        }
    } else {  // PNG / TIFF: 1 value (PNG names it "A": not an alpha), 2 = value + alpha, 3 RGB, 4 RGBA
        if (nc == 1 || nc == 2) pick = {0};
        else if (nc >= 3) pick = {0, 1, 2};
        if (nc == 2) alpha = 1;
        if (nc == 4) alpha = 3;
    }
    if (pick.empty()) Fail(Kind::Channels, "the image has " + std::to_string(nc) + " channels");

    const bool isFloat = src.type == PixelType::F16 || src.type == PixelType::F32;
    d.floatSource = isFloat;
    const float maxCode = src.type == PixelType::U8 ? 255.f : src.type == PixelType::U16 ? 65535.f : 1.f;
    const bool invert = p.format == ImageProbe::Format::Tiff && p.tiffPhotometric == kMinIsWhite && pick.size() == 1;
    const uint32_t w = src.width, h = src.height;
    if (alpha >= 0)
        for (uint32_t y = 0; y < h && !d.alphaIgnored; ++y)
            for (uint32_t x = 0; x < w; ++x)
                if (!(src.Get(x, y, static_cast<size_t>(alpha)) >= maxCode)) {
                    d.alphaIgnored = true;
                    break;
                }
    // grey: one value, or R = G = B exactly in the source's codes everywhere
    bool grey = pick.size() == 1;
    if (!grey) {
        grey = true;
        for (uint32_t y = 0; y < h && grey; ++y)
            for (uint32_t x = 0; x < w; ++x) {
                const float r = src.Get(x, y, pick[0]);
                if (!(r == src.Get(x, y, pick[1]) && r == src.Get(x, y, pick[2]))) {
                    grey = false;
                    break;
                }
            }
    }
    d.grayscale = grey;
    auto toU16 = [&](float v) -> uint16_t {
        if (invert) v = maxCode - v;
        if (src.type == PixelType::U8) return static_cast<uint16_t>(static_cast<uint32_t>(v) * 257u);
        if (src.type == PixelType::U16) return static_cast<uint16_t>(v);
        if (!(v >= 0.f && v <= 1.f)) {  // NaN, +-Inf and out of range: clamped and counted
            ++d.clamped;
            v = std::isnan(v) ? 0.f : std::clamp(v, 0.f, 1.f);
        }
        return static_cast<uint16_t>(std::lround(static_cast<double>(v) * 65535.0));
    };
    const size_t outC = grey ? 1 : 3;
    d.img.Allocate(w, h, PixelType::U16, grey ? std::vector<std::string>{"Y"} : std::vector<std::string>{"R", "G", "B"});
    uint16_t* o = d.img.As<uint16_t>();
    for (uint32_t y = 0; y < h; ++y)
        for (uint32_t x = 0; x < w; ++x)
            for (size_t c = 0; c < outC; ++c) o[(static_cast<size_t>(y) * w + x) * outC + c] = toU16(src.Get(x, y, pick[c]));
    if (!grey)
        for (size_t i = 0; i < static_cast<size_t>(w) * h && !d.bandsDiffer; ++i) d.bandsDiffer = o[i * 3 + 1] != o[i * 3 + 2];
    return d;
}

// "The same content": the stored plane, whatever file it came from (a PNG8 and its PNG16 copy are the same mask).
std::string PixelsSha256(const PassImage& img) {
    const nlohmann::json head = {{"width", img.width},
                                 {"height", img.height},
                                 {"type", std::string(ToString(img.type))},
                                 {"channels", img.channels},
                                 {"data", Sha256Hex(img.data.data(), img.data.size())}};
    const std::string canonical = CanonicalJson(head);
    return "sha256:" + Sha256Hex(canonical.data(), canonical.size());
}

ControlMaskInfo InfoOf(const Manifest& m) {
    ControlMaskInfo i;
    const nlohmann::json& sp = m.stageParams;
    i.fingerprint = m.fingerprint;
    i.created = m.created;
    i.width = m.width;
    i.height = m.height;
    i.grayscale = m.channels.size() == 1;
    if (sp.is_object()) {
        i.sourceName = sp.value("source_name", "");
        i.pixelsSha256 = sp.value("pixels_sha256", "");
        i.bandsDiffer = sp.value("bands_differ", false);
        i.alphaIgnored = sp.value("alpha_ignored", false);
        i.clamped = sp.value("clamped", int64_t{0});
    }
    return i;
}

// One axis of the resize: m output samples from n, each a list of (source index, weight) - a copy when equal, area means where
// the axis shrinks, bilinear with pixel centres (i + 0.5) n / m - 0.5 where it grows (as a GPU samples).
std::vector<std::vector<std::pair<uint32_t, double>>> AxisTaps(uint32_t n, uint32_t m) {
    std::vector<std::vector<std::pair<uint32_t, double>>> taps(m);
    for (uint32_t i = 0; i < m; ++i) {
        if (n == m) {
            taps[i] = {{i, 1.0}};
        } else if (m < n) {
            const double s = static_cast<double>(n) / m, a = i * s, b = (i + 1) * s;
            for (uint32_t j = static_cast<uint32_t>(std::floor(a)); j < n && j < b; ++j) {
                const double overlap = std::min(b, j + 1.0) - std::max(a, static_cast<double>(j));
                if (overlap > 0.0) taps[i].push_back({j, overlap / s});
            }
        } else {
            const double x = std::clamp((i + 0.5) * n / m - 0.5, 0.0, static_cast<double>(n - 1));
            const uint32_t j0 = static_cast<uint32_t>(std::floor(x)), j1 = std::min(j0 + 1, n - 1);
            const double f = x - j0;
            taps[i].push_back({j0, 1.0 - f});
            if (f > 0.0) taps[i].push_back({j1, f});
        }
    }
    return taps;
}

}  // namespace

ControlMaskImport ImportControlMask(const fs::path& image, const fs::path& root) {
    if (root.empty()) Fail(Kind::NoRoot, "no passes root: the control mask lives in the project's passes folder (open a video first)");
    std::error_code ec;
    if (!fs::is_regular_file(image, ec)) Fail(Kind::NotAFile, "not a file: " + Utf8(image));
    Decoded d = DecodeMask(image);
    fs::create_directories(root, ec);
    if (ec) FailFs(ec, "cannot create the passes root " + Utf8(root));
    const std::string pixels = PixelsSha256(d.img);

    ControlMaskImport r;
    r.sourceFormat = std::string(ToString(d.probe.format));
    r.sourceType = std::string(ToString(d.probe.type));
    r.sourceChannels = d.probe.channels;
    r.exrPadded = d.exrPadded;
    ImportLock lock(root);
    CleanLeftovers(root);
    const fs::path cur = root / kControlMaskPass;
    if (auto now = ReadControlMaskInfo(cur); now && now->pixelsSha256 == pixels) {  // these pixels are loaded: nothing to write
        r.outcome = ControlMaskImport::Outcome::Unchanged;
        r.mask = *now;
        return r;
    }

    // the new folder in a working folder of this import, the manifest last; then the current mask out of the way and the new in
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    const fs::path work = root / kImportDir / (std::to_string(_getpid()) + "-" + std::to_string(ms));
    const fs::path tmp = work / kControlMaskPass;
    struct WorkFolder {
        fs::path dir;
        ~WorkFolder() {
            std::error_code e;
            fs::remove_all(dir, e);
        }
    } workFolder{work};
    fs::create_directories(tmp, ec);
    if (ec) FailFs(ec, "cannot create " + Utf8(tmp));
    Manifest m;
    m.pass = kControlMaskPass;
    m.width = d.img.width;
    m.height = d.img.height;
    m.fps = {0, 1};  // no rate: the viewport's base rate is the video's
    m.frameCount = 1;
    m.firstFrame = m.lastFrame = 0;
    m.format = FileFormat::Png;
    m.filePattern = std::string(kControlMaskPass) + "_%06d.png";
    m.pixelType = PixelType::U16;
    m.channels = d.img.channels;
    m.colorspace = "";
    m.convention = Convention::Raw;
    m.stageParams = {{"static", true},
                     {"source_name", Utf8(image.filename())},
                     {"source_sha256", "sha256:" + Sha256File(image)},
                     {"source_format", r.sourceFormat},
                     {"source_type", r.sourceType},
                     {"source_size", {d.probe.width, d.probe.height}},
                     {"grayscale", d.grayscale},
                     {"bands_differ", d.bandsDiffer},
                     {"pixels_sha256", pixels}};
    if (d.alphaIgnored) m.stageParams["alpha_ignored"] = true;
    if (d.floatSource) m.stageParams["clamped"] = d.clamped;
    if (d.exrPadded) m.stageParams["exr_padded"] = true;
    try {
        WritePng(tmp / m.FrameFileName(0), d.img);
        m.fingerprint = ContentFingerprint(tmp, m);
        m.created = NowIso8601();
        m.Save(tmp);
    } catch (const std::exception& e) {
        Fail(Kind::Write, std::string("cannot write the control mask: ") + e.what());
    }

    std::string retired;
    if (fs::is_directory(cur, ec)) {
        r.outcome = ControlMaskImport::Outcome::Replaced;
        std::error_code rec;
        if (ManifestLoads(cur)) {
            retired = RetirePassVersion(root, kControlMaskPass, rec).value_or("");
            if (rec) FailFs(rec, "cannot move the current control mask into the history (it stays in place)");
        } else {  // no loadable manifest: the history would hide it (_broken / _partial), so it is deleted
            fs::remove_all(cur, rec);
            if (rec) FailFs(rec, "cannot remove the broken control mask folder " + Utf8(cur));
        }
    }
    if (ControlMaskHooksForTests().beforePlace) ControlMaskHooksForTests().beforePlace(tmp);
    std::error_code mv;
    fs::rename(tmp, cur, mv);
    if (mv) {
        if (!retired.empty()) {
            try {
                UsePassVersion(root, kControlMaskPass, retired);
            } catch (const std::exception&) {
                throw ControlMaskError(ControlMaskIssue{Kind::Write,
                                                        "cannot put the new control mask in place (" + mv.message() + ") nor bring the previous one back: there is "
                                                        "no mask now, the previous one is in the history " + retired,
                                                        0, 0, false});
            }
        }
        // the previous mask is back in place - or there was none, or a broken folder was deleted at the replace step: no mask now
        FailFs(mv, "cannot put the new control mask in place", !retired.empty());
    }
    r.retiredVersion = retired;
    r.mask = InfoOf(m);
    Log()->info("mask_control: {} {}x{} {} -> {} ({})", r.mask.sourceName, m.width, m.height, d.grayscale ? "grey" : "RGB", Utf8(cur),
                r.outcome == ControlMaskImport::Outcome::Replaced ? "replaced" + (retired.empty() ? std::string() : ", previous " + retired) : std::string("imported"));
    return r;
}

std::optional<std::string> RemoveControlMask(const fs::path& root) {
    if (root.empty()) Fail(Kind::NoRoot, "no passes root");
    const fs::path cur = root / kControlMaskPass;
    std::error_code ec;
    if (!fs::is_directory(cur, ec)) return std::nullopt;
    ImportLock lock(root);
    if (!ManifestLoads(cur)) {
        fs::remove_all(cur, ec);
        if (ec) FailFs(ec, "cannot remove the broken control mask folder " + Utf8(cur));
        Log()->info("mask_control: removed the folder without a loadable manifest {}", Utf8(cur));
        return std::nullopt;
    }
    std::optional<std::string> id = RetirePassVersion(root, kControlMaskPass, ec);
    if (ec) FailFs(ec, "cannot move the control mask into the history");
    return id;
}

bool ControlMaskFolderBroken(const fs::path& root) {
    std::error_code ec;
    return !root.empty() && fs::is_directory(root / kControlMaskPass, ec) && !ManifestLoads(root / kControlMaskPass);
}

std::optional<ControlMaskInfo> ReadControlMaskInfo(const fs::path& dir, std::string* why) {
    auto no = [&](std::string w) -> std::optional<ControlMaskInfo> {
        if (why) *why = std::move(w);
        return std::nullopt;
    };
    if (!Manifest::Exists(dir)) return no("absent");
    Manifest m;
    try {
        m = Manifest::Load(dir);
    } catch (const std::exception& e) {
        return no(std::string("unreadable: ") + e.what());
    }
    if (m.pass != kControlMaskPass) return no("not a mask_control pass (pass '" + m.pass + "')");
    if (!m.IsStatic()) return no("not static one-frame");
    if (m.channels != std::vector<std::string>{"Y"} && m.channels != std::vector<std::string>{"R", "G", "B"}) {
        std::string names;
        for (const auto& c : m.channels) names += (names.empty() ? "" : ",") + c;
        return no("channels " + names + " (needs Y or R,G,B)");
    }
    if (m.format != FileFormat::Png || m.pixelType != PixelType::U16)
        return no("type " + std::string(ToString(m.format)) + " " + std::string(ToString(m.pixelType)) + " (needs png u16)");
    if (m.fingerprint.empty()) return no("not stamped");
    try {
        if (ContentFingerprint(dir, m, true) != m.fingerprint) return no("changed outside dlssvid");
    } catch (const std::exception&) {
        return no("changed outside dlssvid");  // the frame file is gone
    }
    return InfoOf(m);
}

ControlMaskTexels LoadControlMaskForFrame(const fs::path& dir, uint32_t w, uint32_t h, const std::string& expectFingerprint) {
    if (w == 0 || h == 0) Throw("mask_control: the frame has no pixels");
    const Manifest m = Manifest::Load(dir);
    const PassImage img = ReadPng(dir / m.FrameFileName(0));
    if (img.type != PixelType::U16 || (img.channels.size() != 1 && img.channels.size() != 3)) Throw("mask_control: the frame file is not a png16 Y / RGB image");
    // decoded, then checked: pixels of a mask replaced since Init never get the fingerprint Init read (the plan's stamp)
    if (ContentFingerprint(dir, m, true) != expectFingerprint) Throw("the control mask changed while nr started: run again");

    const size_t nc = img.channels.size();
    const uint32_t sw = img.width, sh = img.height;
    const auto tx = AxisTaps(sw, w), ty = AxisTaps(sh, h);
    const uint16_t* src = img.As<uint16_t>();
    // the axis whose pass leaves the smaller intermediate goes first
    const bool xFirst = static_cast<uint64_t>(sh) * w <= static_cast<uint64_t>(h) * sw;
    std::vector<float> mid;
    ControlMaskTexels t;
    t.width = w;
    t.height = h;
    t.rgb = nc == 3;
    const size_t outC = t.rgb ? 4 : 1;
    t.data.assign(static_cast<size_t>(w) * h * outC, uint16_t{65535});
    auto put = [&](uint32_t x, uint32_t y, size_t c, double v) {
        t.data[(static_cast<size_t>(y) * w + x) * outC + c] = static_cast<uint16_t>(std::clamp(std::lround(v), 0L, 65535L));
    };
    if (xFirst) {
        mid.resize(static_cast<size_t>(sh) * w * nc);
        for (uint32_t y = 0; y < sh; ++y)
            for (uint32_t x = 0; x < w; ++x)
                for (size_t c = 0; c < nc; ++c) {
                    double v = 0.0;
                    for (const auto& [j, wt] : tx[x]) v += wt * src[(static_cast<size_t>(y) * sw + j) * nc + c];
                    mid[(static_cast<size_t>(y) * w + x) * nc + c] = static_cast<float>(v);
                }
        for (uint32_t y = 0; y < h; ++y)
            for (uint32_t x = 0; x < w; ++x)
                for (size_t c = 0; c < nc; ++c) {
                    double v = 0.0;
                    for (const auto& [j, wt] : ty[y]) v += wt * mid[(static_cast<size_t>(j) * w + x) * nc + c];
                    put(x, y, c, v);
                }
    } else {
        mid.resize(static_cast<size_t>(h) * sw * nc);
        for (uint32_t y = 0; y < h; ++y)
            for (uint32_t x = 0; x < sw; ++x)
                for (size_t c = 0; c < nc; ++c) {
                    double v = 0.0;
                    for (const auto& [j, wt] : ty[y]) v += wt * src[(static_cast<size_t>(j) * sw + x) * nc + c];
                    mid[(static_cast<size_t>(y) * sw + x) * nc + c] = static_cast<float>(v);
                }
        for (uint32_t y = 0; y < h; ++y)
            for (uint32_t x = 0; x < w; ++x)
                for (size_t c = 0; c < nc; ++c) {
                    double v = 0.0;
                    for (const auto& [j, wt] : tx[x]) v += wt * mid[(static_cast<size_t>(y) * sw + j) * nc + c];
                    put(x, y, c, v);
                }
    }
    if (t.rgb)
        for (size_t i = 0; i < static_cast<size_t>(w) * h && !t.split; ++i) t.split = t.data[i * 4 + 1] != t.data[i * 4 + 2];
    return t;
}

std::string ControlMaskLabelOf(const Manifest& nrPass) {
    const nlohmann::json& sp = nrPass.stageParams;
    if (sp.is_object() && sp.contains("control_mask") && sp["control_mask"].is_object()) return sp["control_mask"].value("fingerprint", "");
    return {};
}

}  // namespace dlssvid
