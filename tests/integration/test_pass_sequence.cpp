// Stage 1 acceptance (ТЗ §8): export -> import round trip without loss, presets, validation.

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>

#include "TestClips.h"
#include "convert/ColorConvert.h"
#include "convert/DepthConvert.h"
#include "convert/MvConvert.h"
#include "io/VideoDecoder.h"
#include "passes/PassFile.h"
#include "passes/PassSequence.h"
#include "passes/formats/ExrIO.h"

using namespace dlssvid;
using namespace dlssvid::test;

namespace {

std::filesystem::path Dir(const char* name) {
    const auto d = TempDir() / "passes" / name;
    std::filesystem::remove_all(d);
    return d;
}

PassImage SyntheticDepth(int64_t frame, uint32_t w, uint32_t h) {
    PassImage d = MakePassImage(PassKind::DepthRaw, w, h);
    for (uint32_t y = 0; y < h; ++y)
        for (uint32_t x = 0; x < w; ++x) d.Set(x, y, 0, 1.f + 0.01f * x + 0.02f * y + 0.1f * static_cast<float>(frame) + 1e-3f * std::sin(x * 0.3f));
    return d;
}

PassImage SyntheticMv(int64_t frame, uint32_t w, uint32_t h) {
    PassImage m = MakePassImage(PassKind::MvRaw, w, h);
    for (uint32_t y = 0; y < h; ++y)
        for (uint32_t x = 0; x < w; ++x) {
            m.Set(x, y, 0, 0.5f * static_cast<float>(frame) - 0.001f * x);
            m.Set(x, y, 1, -0.25f + 0.002f * y);
        }
    return m;
}

FrameSource DepthSource(uint32_t w, uint32_t h, int64_t count) {
    return [=](int64_t f, PassImage& out) {
        if (f < 0 || f >= count) return false;
        out = SyntheticDepth(f, w, h);
        return true;
    };
}

}  // namespace

TEST_CASE("Export -> import of depth_raw as EXR is lossless and the manifest describes it", "[integration][passes]") {
    const auto dir = Dir("depth_exr");
    Manifest m = Manifest::ForPass(PassKind::DepthRaw, 64, 40, FileFormat::Exr);
    m.fps = Rational{24, 1};
    m.model = "test";
    int64_t progressCalls = 0;
    const int64_t n = ExportPass(dir, m, DepthSource(64, 40, 10), FrameRange{0, -1}, [&](int64_t, int64_t) { ++progressCalls; });
    CHECK(n == 10);
    CHECK(progressCalls == 10);
    REQUIRE(Manifest::Exists(dir));

    const PassReader r = PassReader::Open(dir);
    CHECK(r.Man().frameCount == 10);
    CHECK(r.Man().firstFrame == 0);
    CHECK(r.Man().lastFrame == 9);
    CHECK(r.Man().pass == "depth_raw");
    CHECK(r.Man().fps.num == 24);
    CHECK(r.MissingFrames().empty());
    for (int64_t f = 0; f < 10; ++f) {
        const PassImage back = r.ReadFrame(f);
        const PassImage ref = SyntheticDepth(f, 64, 40);
        REQUIRE(back.SameLayout(ref));
        REQUIRE(back.data == ref.data);
    }
    r.Validate({64, 40, 10, PassKind::DepthRaw});
    CHECK_THROWS(r.Validate({65, 40, 10, std::nullopt}));
    CHECK_THROWS(r.Validate({0, 0, 9, std::nullopt}));
    CHECK_THROWS(r.Validate({0, 0, -1, PassKind::MvRaw}));
}

TEST_CASE("Every format round-trips depth and mv through a pass folder", "[integration][passes]") {
    for (FileFormat fmt : {FileFormat::Exr, FileFormat::Tiff, FileFormat::Npz, FileFormat::Raw}) {
        for (PassKind kind : {PassKind::DepthRaw, PassKind::MvRaw, PassKind::DepthDlss, PassKind::MvDlss}) {
            const auto dir = Dir("fmt");
            Manifest m = Manifest::ForPass(kind, 24, 16, fmt);
            m.pixelType = PixelType::F32;  // exact round trip for all containers
            FrameSource src = [kind](int64_t f, PassImage& out) {
                if (f >= 3) return false;
                out = kind == PassKind::DepthRaw || kind == PassKind::DepthDlss ? SyntheticDepth(f, 24, 16) : SyntheticMv(f, 24, 16);
                out.channels = Spec(kind).channels;
                return true;
            };
            CHECK(ExportPass(dir, m, src, FrameRange{0, 2}) == 3);
            const PassReader r = PassReader::Open(dir);
            for (int64_t f = 0; f < 3; ++f) {
                PassImage ref;
                src(f, ref);
                const PassImage back = r.ReadFrame(f);
                REQUIRE(back.SameLayout(ref));
                REQUIRE(back.data == ref.data);
            }
        }
    }
}

TEST_CASE("Export range, missing frames and scanning without a manifest", "[integration][passes]") {
    const auto dir = Dir("range");
    Manifest m = Manifest::ForPass(PassKind::DepthRaw, 20, 10, FileFormat::Exr);
    CHECK(ExportPass(dir, m, DepthSource(20, 10, 100), FrameRange{5, 9}) == 5);
    CHECK_THROWS(ExportPass(Dir("range2"), m, DepthSource(20, 10, 3), FrameRange{0, 9}));  // source too short for explicit range
    PassReader r = PassReader::Open(dir);
    CHECK(r.Man().firstFrame == 5);
    CHECK(r.Man().lastFrame == 9);
    std::filesystem::remove(r.FramePath(7));
    CHECK(r.MissingFrames() == std::vector<int64_t>{7});
    CHECK_THROWS(r.Validate({}));
    CHECK_THROWS(r.ReadFrame(7));

    std::filesystem::remove(dir / Manifest::kFileName);
    CHECK_THROWS(PassReader::Open(dir));
    PassReader::ScanHints hints;
    hints.kind = PassKind::DepthRaw;
    const PassReader s = PassReader::OpenWithoutManifest(dir, hints);
    CHECK_FALSE(s.HadManifest());
    CHECK(s.Man().width == 20);
    CHECK(s.Man().height == 10);
    CHECK(s.Man().firstFrame == 5);
    CHECK(s.Man().lastFrame == 9);
    CHECK(s.Man().frameCount == 4);
    CHECK(s.ReadFrame(6).data == SyntheticDepth(6, 20, 10).data);
    CHECK(s.MissingFrames() == std::vector<int64_t>{7});

    CHECK(FrameRange::Parse("3-7")->last == 7);
    CHECK(FrameRange::Parse("3-")->last == -1);
    CHECK(FrameRange::Parse("4")->first == 4);
    CHECK_FALSE(FrameRange::Parse("x"));
    CHECK_FALSE(FrameRange::Parse("7-3"));
}

TEST_CASE("Colour export from a clip round-trips as EXR half and PNG16", "[integration][passes][color]") {
    ClipSpec spec;
    spec.frames = 6;
    spec.width = 64;
    spec.height = 36;
    const auto clip = WriteClip(TempDir() / "passes_color.mkv", spec);
    VideoDecoder dec(clip);
    const ColorInfo ci = ColorInfoFromStream(dec.Info());
    std::vector<PassImage> frames;
    CpuFrame f;
    while (dec.NextFrame(f)) frames.push_back(YuvToRgb(f, ci, PixelType::F16));
    REQUIRE(frames.size() == 6);

    FrameSource src = [&](int64_t i, PassImage& out) {
        if (i < 0 || i >= static_cast<int64_t>(frames.size())) return false;
        out = frames[static_cast<size_t>(i)];
        return true;
    };
    const auto exr = Dir("color_exr");
    Manifest m = Manifest::ForPass(PassKind::ColorSource, 64, 36, FileFormat::Exr);
    CHECK(ExportPass(exr, m, src, FrameRange{}) == 6);
    const PassReader r = PassReader::Open(exr);
    for (int64_t i = 0; i < 6; ++i) REQUIRE(r.ReadFrame(i).data == frames[static_cast<size_t>(i)].data);

    // PNG16 from the half frames as the stages write them (TASK-0042: they came out as 0 / 1 of 65535 - black):
    // [0,1] scaled to the full range by the writer, reversible within 1/65535
    const auto png = Dir("color_png");
    Manifest p = Manifest::ForPass(PassKind::ColorSource, 64, 36, FileFormat::Png);
    CHECK(ExportPass(png, p, src, FrameRange{}) == 6);
    const PassReader rp = PassReader::Open(png);
    CHECK(rp.Man().pixelType == PixelType::U16);
    const PassImage b = rp.ReadFrame(2);
    CHECK(b.channels == std::vector<std::string>{"R", "G", "B"});
    double maxErr = 0.0;
    for (uint32_t y = 0; y < 36; ++y)
        for (uint32_t x = 0; x < 64; ++x)
            for (size_t c = 0; c < 3; ++c) maxErr = std::max(maxErr, std::fabs(b.Get(x, y, c) / 65535.0 - frames[2].Get(x, y, c)));
    CHECK(maxErr <= 1.0 / 65535.0);

    // and back: PNG16 -> EXR half gives the same colour (was 0..65535 in the half file)
    const auto exr2 = Dir("color_png_exr");
    CHECK(ExportPass(exr2, Manifest::ForPass(PassKind::ColorSource, 64, 36, FileFormat::Exr), SourceFromReader(rp), FrameRange{}) == 6);
    const PassImage e2 = PassReader::Open(exr2).ReadFrame(2);
    REQUIRE(e2.type == PixelType::F16);
    maxErr = 0.0;
    for (uint32_t y = 0; y < 36; ++y)
        for (uint32_t x = 0; x < 64; ++x)
            for (size_t c = 0; c < 3; ++c) maxErr = std::max(maxErr, std::fabs(static_cast<double>(e2.Get(x, y, c)) - frames[2].Get(x, y, c)));
    CHECK(maxErr <= 1e-3);  // half rounding

    // a mask written from float samples: 1.0 -> 255
    const auto mask = Dir("mask_png");
    {
        PassWriter w(mask, Manifest::ForPass(PassKind::Mask, 4, 2, FileFormat::Png));
        PassImage mf = MakePassImage(PassKind::Mask, 4, 2, PixelType::F32);
        for (uint32_t x = 0; x < 4; ++x) mf.Set(x, 0, 0, 0.25f * static_cast<float>(x + 1));
        w.WriteFrame(0, mf);
        w.Finish();
    }
    const PassImage mr = PassReader::Open(mask).ReadFrame(0);
    REQUIRE(mr.type == PixelType::U8);
    CHECK(mr.Get(3, 0, 0) == 255.f);
    CHECK(mr.Get(1, 0, 0) == 128.f);
}

// Review of TASK-0042: a folder of half EXR mattes (a roto from Nuke) without a manifest got a u8 manifest, and the EXR
// restore rounded 0.5 to 0 or 1 - an empty mask.
TEST_CASE("Float EXR masks scanned without a manifest keep their [0,1] values", "[integration][passes][regression]") {
    const auto dir = Dir("mask_exr_scan");
    std::filesystem::create_directories(dir);
    PassImage m;
    m.Allocate(4, 2, PixelType::F16, {"A"});
    for (uint32_t x = 0; x < 4; ++x) m.Set(x, 0, 0, 0.25f * static_cast<float>(x + 1));
    WritePassFile(dir / "matte_000000.exr", m);
    PassReader::ScanHints hints;
    hints.kind = PassKind::Mask;
    const PassReader r = PassReader::OpenWithoutManifest(dir, hints);
    CHECK(r.Man().pixelType == PixelType::F16);
    const PassImage back = r.ReadFrame(0);
    CHECK(back.Get(1, 0, 0) == 0.5f);  // before: 0 or 1
    CHECK(back.Get(3, 0, 0) == 1.f);
}

TEST_CASE("Nuke preset writes multi-layer EXR with colour, depth and mv", "[integration][passes][preset]") {
    const uint32_t w = 32, h = 20;
    std::vector<PassImage> color(4);
    for (int i = 0; i < 4; ++i) {
        color[i] = MakePassImage(PassKind::ColorSource, w, h);
        for (uint32_t y = 0; y < h; ++y)
            for (uint32_t x = 0; x < w; ++x)
                for (size_t c = 0; c < 3; ++c) color[i].Set(x, y, c, (x + y + i * 3 + c) / 80.f);
    }
    FrameSource cs = [&](int64_t i, PassImage& out) {
        if (i >= 4) return false;
        out = color[static_cast<size_t>(i)];
        return true;
    };
    FrameSource ds = DepthSource(w, h, 4);
    FrameSource ms = [&](int64_t i, PassImage& out) {
        if (i >= 4) return false;
        out = SyntheticMv(i, w, h);
        return true;
    };
    const auto dir = Dir("nuke");
    Manifest base = Manifest::ForPass(PassKind::ColorSource, w, h, FileFormat::Exr);
    base.fps = Rational{25, 1};
    CHECK(ExportLayeredExr(dir, "layered", base, cs, &ds, &ms, FrameRange{}) == 4);
    const Manifest m = Manifest::Load(dir);
    CHECK(m.channels == std::vector<std::string>{"R", "G", "B", "depth.Z", "mv.u", "mv.v"});
    CHECK(m.frameCount == 4);
    const auto layers = ReadExrLayers(dir / m.FrameFileName(2));
    REQUIRE(layers.size() == 3);
    CHECK(layers.at("").data == color[2].data);
    CHECK(layers.at("depth").data == SyntheticDepth(2, w, h).data);
    CHECK(layers.at("mv").data == SyntheticMv(2, w, h).data);

    // colour from a PNG16 pass (u16, full range) lands in the colour layer as [0,1] (TASK-0042: was 0..65535)
    const auto png = Dir("nuke_png_src");
    CHECK(ExportPass(png, Manifest::ForPass(PassKind::ColorSource, w, h, FileFormat::Png), cs, FrameRange{}) == 4);
    const PassReader rp = PassReader::Open(png);
    const auto dir2 = Dir("nuke_from_png");
    CHECK(ExportLayeredExr(dir2, "layered", base, SourceFromReader(rp), &ds, &ms, FrameRange{}) == 4);
    const auto layers2 = ReadExrLayers(dir2 / m.FrameFileName(2));
    const PassImage& c2 = layers2.at("");
    REQUIRE(c2.width == w);
    REQUIRE(c2.ChannelCount() == 3);
    float maxErr = 0.f;
    for (uint32_t y = 0; y < h; ++y)
        for (uint32_t x = 0; x < w; ++x)
            for (size_t c = 0; c < 3; ++c) maxErr = std::max(maxErr, std::fabs(c2.Get(x, y, c) - color[2].Get(x, y, c)));
    CHECK(maxErr <= 1e-3f);
}

TEST_CASE("Preset formats: comfyui and rawdlss", "[integration][passes][preset]") {
    CHECK(PresetFormat(ExportPreset::ComfyUi, PassKind::ColorSource) == FileFormat::Png);
    CHECK(PresetPixelType(ExportPreset::ComfyUi, PassKind::ColorSource) == PixelType::U16);
    CHECK(PresetFormat(ExportPreset::ComfyUi, PassKind::DepthRaw) == FileFormat::Npz);
    CHECK(PresetFormat(ExportPreset::RawDlss, PassKind::DepthDlss) == FileFormat::Raw);
    CHECK(PresetPixelType(ExportPreset::RawDlss, PassKind::MvDlss) == PixelType::F16);
    CHECK(ParseExportPreset("nuke") == ExportPreset::Nuke);
    CHECK(ParseExportPreset("") == ExportPreset::None);
    CHECK_FALSE(ParseExportPreset("bogus"));

    // rawdlss: depth_dlss as .r32, readable back with the manifest
    const auto dir = Dir("rawdlss");
    Manifest m = Manifest::ForPass(PassKind::DepthDlss, 16, 8, FileFormat::Raw);
    DepthParams p;
    FrameSource src = [&](int64_t f, PassImage& out) {
        if (f >= 2) return false;
        DepthParams local = p;
        out = DepthRawToDlss(SyntheticDepth(f, 16, 8), local);
        return true;
    };
    CHECK(ExportPass(dir, m, src, FrameRange{}) == 2);
    CHECK(std::filesystem::exists(dir / "depth_dlss_000001.r32"));
    CHECK(std::filesystem::file_size(dir / "depth_dlss_000001.r32") == 16 * 8 * 4);
    const PassReader r = PassReader::Open(dir);
    PassImage ref;
    src(1, ref);
    CHECK(r.ReadFrame(1).data == ref.data);
}

// Audit pass-mv-yup (TASK-0038): a pass stored with y up (manifest mv.y_up, another tool's convention) comes out of
// PassReader::ReadFrame in the DLSS convention (y down) for every consumer; the manifest keeps describing the files.
namespace {
void MarkYUp(const std::filesystem::path& dir) {  // the way a foreign tool or `import --y-up --write-manifest` marks it
    Manifest m = Manifest::Load(dir);
    m.mv.yUp = true;
    m.Save(dir);
}
}  // namespace

TEST_CASE("PassReader returns y_up motion vectors in the DLSS convention; the manifest keeps the stored flag",
          "[integration][passes][mv][y_up]") {
    SECTION("every format: frames come back y down, the files and the manifest keep y up") {
        struct Case {
            FileFormat format;
            PassKind kind;
            PixelType type;
        };
        for (const Case c : {Case{FileFormat::Exr, PassKind::MvRaw, PixelType::F32}, Case{FileFormat::Raw, PassKind::MvDlss, PixelType::F16},
                             Case{FileFormat::Npz, PassKind::MvRaw, PixelType::F32}}) {
            INFO(ToString(c.format) << " " << ToString(c.kind) << " " << ToString(c.type));
            const auto a = Dir("yup_a");
            const auto b = Dir("yup_b");
            Manifest m = Manifest::ForPass(c.kind, 24, 16, c.format);
            m.pixelType = c.type;
            auto source = [&](bool flipped) {
                return FrameSource([=](int64_t f, PassImage& out) {
                    if (f >= 3) return false;
                    out = SyntheticMv(f, 24, 16);
                    out.channels = Spec(c.kind).channels;
                    if (flipped) out = FlipMvY(out);
                    return true;
                });
            };
            REQUIRE(ExportPass(a, m, source(false), FrameRange{0, 2}) == 3);
            Manifest yUp = m;
            yUp.mv.yUp = true;  // a writer never stores the flag: frames reaching it are y down
            REQUIRE(ExportPass(b, yUp, source(true), FrameRange{0, 2}) == 3);
            CHECK_FALSE(Manifest::Load(b).mv.yUp);
            MarkYUp(b);

            const PassReader ra = PassReader::Open(a);
            const PassReader rb = PassReader::Open(b);
            CHECK(rb.Man().mv.yUp);
            CHECK_FALSE(rb.FrameManifest().mv.yUp);
            rb.Validate({24, 16, 3, c.kind});
            PassImage viaSource;
            const FrameSource fromB = SourceFromReader(rb);
            for (int64_t f = 0; f < 3; ++f) {
                const PassImage want = ra.ReadFrame(f);
                const PassImage got = rb.ReadFrame(f);
                REQUIRE(got.SameLayout(want));
                CHECK(got.data == want.data);  // negated back exactly
                CHECK(got.Get(3, 5, 1) == want.Get(3, 5, 1));
                CHECK(got.Get(3, 5, 1) != 0.f);
                // the file level is untouched: ReadPassFile returns what is stored (v pointing up)
                const PassImage stored = ReadPassFile(rb.FramePath(f), &rb.Man());
                CHECK(stored.Get(3, 5, 1) == -want.Get(3, 5, 1));
                REQUIRE(fromB(f, viaSource));
                CHECK(viaSource.data == want.data);
            }
        }
    }

    SECTION("y_up on a non-MV pass changes nothing") {
        const auto d = Dir("yup_depth");
        REQUIRE(ExportPass(d, Manifest::ForPass(PassKind::DepthRaw, 24, 16, FileFormat::Exr), DepthSource(24, 16, 2), FrameRange{0, 1}) == 2);
        MarkYUp(d);
        CHECK(PassReader::Open(d).ReadFrame(1).data == SyntheticDepth(1, 24, 16).data);
    }

    SECTION("y-up motion vectors in integer samples are refused at Validate and on read") {
        const auto d = Dir("yup_png16");
        Manifest m = Manifest::ForPass(PassKind::MvRaw, 24, 16, FileFormat::Png);
        m.pixelType = PixelType::U16;
        FrameSource positive = [](int64_t f, PassImage& out) {
            if (f >= 1) return false;
            out = MakePassImage(PassKind::MvRaw, 24, 16, PixelType::U16);
            return true;
        };
        REQUIRE(ExportPass(d, m, positive, FrameRange{0, 0}) == 1);
        MarkYUp(d);
        const PassReader r = PassReader::Open(d);
        CHECK_THROWS(r.Validate({}));
        CHECK_THROWS(r.ReadFrame(0));
    }
}
