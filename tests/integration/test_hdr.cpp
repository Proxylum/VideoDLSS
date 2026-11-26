// TASK-0078 (HDR TB-1): what a real file's colour reads as - every layer (container, codec parameters, the first frame or packet), the
// HDR records, Dolby Vision, the first look's cache and `dlssvid info`. CPU cases on FFV1 / ProRes; [gpu] ones on NVENC's HEVC / AV1.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <optional>
#include <string>
#include <vector>

#include "TestClips.h"
#include "gpu/D3D12Device.h"
#include "io/SourceColor.h"
#include "io/VideoDecoder.h"
#include "io/VideoEncoder.h"

extern "C" {
#include <libavcodec/avcodec.h>
}

using namespace dlssvid;
using namespace dlssvid::test;
using Catch::Matchers::ContainsSubstring;

namespace {

std::filesystem::path Dir(const char* name) {
    const auto d = TempDir() / "hdr" / name;
    std::filesystem::remove_all(d);
    std::filesystem::create_directories(d);
    return d;
}

struct CliRun {
    int code = 0;
    std::string out;
};

CliRun RunAll(const std::vector<std::string>& args, const std::filesystem::path& log) {
    std::string cmd = "\"\"" DLSSVID_CLI_PATH "\"";
    for (const auto& a : args) cmd += " \"" + a + "\"";
    cmd += " > \"" + log.string() + "\" 2>&1\"";
    CliRun r;
    r.code = std::system(cmd.c_str());
    std::ifstream in(log, std::ios::binary);
    r.out.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    return r;
}

bool HaveNvidia() {
    D3D12Device dev;
    return !dev.IsWarp() && dev.IsNvidia();
}

RawTags Tags(int m, int p, int t, int range = 1) { return RawTags{m, p, t, range, 0}; }

// BT.2020 primaries, D65, 0.0001-1000 cd/m2 - in MP4 / MOV's units (1/50000, 1/10000): exact through mdcv
Mdcv Mastering1000() {
    Mdcv m;
    m.primaries[0][0] = {35400, 50000};
    m.primaries[0][1] = {14600, 50000};
    m.primaries[1][0] = {8500, 50000};
    m.primaries[1][1] = {39850, 50000};
    m.primaries[2][0] = {6550, 50000};
    m.primaries[2][1] = {2300, 50000};
    m.white[0] = {15635, 50000};
    m.white[1] = {16450, 50000};
    m.minLum = {1, 10000};
    m.maxLum = {10000000, 10000};
    m.hasPrimaries = m.hasLuminance = true;
    return m;
}

// Matroska stores the values as floats (matroskadec reads them back through av_d2q): equal within its rounding
bool Near(const Mdcv& a, const Mdcv& b) {
    auto same = [](const Ratio& x, const Ratio& y) { return std::abs(x.Value() - y.Value()) <= 1e-4 * std::max(1.0, std::abs(y.Value())); };
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 2; ++j)
            if (!same(a.primaries[i][j], b.primaries[i][j])) return false;
    return same(a.white[0], b.white[0]) && same(a.white[1], b.white[1]) && same(a.minLum, b.minLum) && same(a.maxLum, b.maxLum) &&
           a.hasPrimaries == b.hasPrimaries && a.hasLuminance == b.hasLuminance;
}

// A ProRes MOV whose colr says 9/9/14 (BT.2020 SDR) and whose frame headers say HLG - the camera's way and the case FFmpeg's demuxer cannot
// tell (proresdec puts a frame's colour into the frame only)
std::filesystem::path ProResHlg(const std::filesystem::path& dir) {
    ClipSpec spec;
    spec.width = 64;
    spec.height = 48;
    spec.frames = 4;
    spec.codec = "prores_ks";
    spec.pixFmt = "yuv422p10le";
    spec.color = Tags(9, 9, 14);
    spec.frameColor = [](int64_t) { return Tags(9, 9, 18); };
    spec.hdr.mastering = Mastering1000();
    spec.hdr.light = Cll{1000, 400};
    spec.hdr.ambient = Amve{{3140000, 10000}, {15635, 50000}, {16450, 50000}};
    return WriteClip(dir / "hlg.mov", spec);
}

std::filesystem::path FfvPq(const std::filesystem::path& dir, const char* name, const ClipHdr& hdr = {},
                            std::function<std::vector<std::pair<int, std::string>>(int64_t)> sideData = {}) {
    ClipSpec spec;
    spec.width = 64;
    spec.height = 48;
    spec.frames = 4;
    spec.pixFmt = "yuv420p10le";
    spec.color = Tags(9, 9, 16);
    spec.hdr = hdr;
    spec.videoPacketSideData = std::move(sideData);
    return WriteClip(dir / name, spec);
}

void SetOld(const std::filesystem::path& file, std::chrono::minutes ago) {  // settled for the first look's cache
    std::filesystem::last_write_time(file, std::filesystem::file_time_type::clock::now() - ago);
}

}  // namespace

TEST_CASE("ProRes MOV with colr trc 14 and frames 18 is HLG from the frame, mastering MaxCLL and amve from the container",
          "[integration][hdr]") {
    const auto dir = Dir("prores_hlg");
    const auto clip = ProResHlg(dir);
    // what FFmpeg 7.1.1 gives: the codec parameters are the colr box (the frame's colour stays in the frame), the frame says 18
    const StreamSummary s = Summarize(clip);
    REQUIRE(s.colorTrc == 14);
    REQUIRE(s.container.trc == 14);
    REQUIRE(FrameColors(clip).at(0).trc == 18);
    REQUIRE(s.hdr.mdcv);
    REQUIRE(s.hdr.cll);
    REQUIRE(s.hdr.amve);

    const int64_t decodesBefore = FirstFrameDecodes();
    VideoDecoder dec(clip, VideoDecoder::Probe());
    const SourceColor& c = dec.Info().color;
    CHECK(FirstFrameDecodes() == decodesBefore + 1);
    CHECK(c.looked == FirstLook::Decode);
    CHECK(c.cls == SourceClass::Hlg);
    CHECK(c.effective.trc == 18);
    CHECK(c.trcFrom == TagFrom::Frame);
    CHECK(c.primariesFrom == TagFrom::Stream);  // the frame says the same: the stream's
    CHECK(c.matrixFrom == TagFrom::Stream);
    CHECK(c.stream.trc == 14);
    CHECK(dec.Info().colorTrc == 14);  // the legacy fields as before
    REQUIRE(c.hdr.mdcv);
    CHECK(*c.hdr.mdcv == Mastering1000());  // mdcv's units: exact
    CHECK(c.mdcvFrom == TagFrom::Container);  // the frame carries the same (libavcodec copies it): the container's
    REQUIRE(c.hdr.cll);
    CHECK(*c.hdr.cll == Cll{1000, 400});
    CHECK(c.cllFrom == TagFrom::Container);
    REQUIRE(c.hdr.amve);
    CHECK(c.hdr.amve->illuminance.Value() == 314.0);
    CHECK(c.amveFrom == TagFrom::Container);
    CHECK(c.notes.empty());  // an alternative transfer is no news
}

TEST_CASE("FFV1 MKV PQ reads MasteringMetadata and MaxCLL without decoding a frame", "[integration][hdr]") {
    const auto dir = Dir("ffv1_pq");
    ClipHdr hdr;
    hdr.mastering = Mastering1000();
    hdr.light = Cll{1000, 400};
    const auto clip = FfvPq(dir, "pq.mkv", hdr);
    const int64_t decodes = FirstFrameDecodes(), peeks = FirstPacketPeeks();
    VideoDecoder dec(clip, VideoDecoder::Probe());
    const SourceColor& c = dec.Info().color;
    CHECK(FirstFrameDecodes() == decodes);  // FFV1 keeps no colour of its own: nothing to decode
    CHECK(FirstPacketPeeks() == peeks + 1);  // PQ in Matroska: the first packet for HDR10+
    CHECK(c.looked == FirstLook::Packet);
    CHECK(c.cls == SourceClass::Pq);
    CHECK(c.effective == RawTags{9, 9, 16, 1, 0});
    CHECK(c.trcFrom == TagFrom::Stream);
    REQUIRE(c.hdr.mdcv);
    CHECK(Near(*c.hdr.mdcv, Mastering1000()));
    CHECK(c.mdcvFrom == TagFrom::Container);
    REQUIRE(c.hdr.cll);
    CHECK(*c.hdr.cll == Cll{1000, 400});
    CHECK_FALSE(c.hdr.amve);  // Matroska has no field for it
    CHECK_FALSE(c.hdr10Plus);
    CHECK(c.notes.empty());
    CHECK_FALSE(c.legacyPath);
}

TEST_CASE("HDR probe cache decodes once per file and again after a new mtime", "[integration][hdr][cache]") {
    const auto dir = Dir("cache");
    const auto clip = ProResHlg(dir);
    SetOld(clip, std::chrono::minutes(60));
    const int64_t start = FirstFrameDecodes();
    { VideoDecoder dec(clip, VideoDecoder::Probe()); CHECK(dec.Info().color.cls == SourceClass::Hlg); }
    { VideoDecoder dec(clip, VideoDecoder::Probe()); CHECK(dec.Info().color.cls == SourceClass::Hlg); }
    { VideoDecoder dec(clip); CHECK(dec.Info().color.trcFrom == TagFrom::Frame); }
    CHECK(FirstFrameDecodes() == start + 1);
    SetOld(clip, std::chrono::minutes(30));  // another file now
    { VideoDecoder dec(clip, VideoDecoder::Probe()); CHECK(dec.Info().color.cls == SourceClass::Hlg); }
    CHECK(FirstFrameDecodes() == start + 2);
    { VideoDecoder dec(clip, VideoDecoder::Probe()); }
    CHECK(FirstFrameDecodes() == start + 2);
    // a file just written is looked at again: its time may not change with its next write
    const auto fresh = ProResHlg(Dir("cache_fresh"));
    const int64_t before = FirstFrameDecodes();
    { VideoDecoder a(fresh, VideoDecoder::Probe()); }
    { VideoDecoder b(fresh, VideoDecoder::Probe()); }
    CHECK(FirstFrameDecodes() == before + 2);
    // a reader without the colour (the result check, thumbnails) never looks
    SetOld(fresh, std::chrono::minutes(60));
    VideoDecoder::Options off = VideoDecoder::Probe();
    off.colorProbe = false;
    const int64_t quiet = FirstFrameDecodes();
    VideoDecoder dec(fresh, off);
    CHECK(FirstFrameDecodes() == quiet);
    CHECK(dec.Info().color.looked == FirstLook::None);
    CHECK(dec.Info().color.cls == SourceClass::Wide);  // colr 9/9/14 alone
}

TEST_CASE("Dolby Vision mapping in FFV1 MKV: 8.4 is HLG with the note, 5 is marked as an IPT base", "[integration][hdr][dovi]") {
    const auto dir = Dir("dovi");
    auto clip = [&](const char* name, RawTags tags, int profile, int compat) {
        ClipSpec spec;
        spec.width = 64;
        spec.height = 48;
        spec.frames = 3;
        spec.pixFmt = "yuv420p10le";
        spec.color = tags;
        DoviConf d;
        d.major = 1;
        d.profile = profile;
        d.level = 6;
        d.blCompat = compat;
        d.rpu = d.bl = true;
        spec.hdr.dovi = d;
        return WriteClip(dir / name, spec);
    };
    const auto hlg = clip("p84.mkv", Tags(9, 9, 18), 8, 4);
    REQUIRE(Summarize(hlg).dovi);  // matroskaenc wrote the mapping, matroskadec read it back
    CHECK(Summarize(hlg).dovi->blCompat == 4);
    VideoDecoder p84(hlg, VideoDecoder::Probe());
    const SourceColor& a = p84.Info().color;
    REQUIRE(a.doviConf);
    CHECK(a.doviConf->profile == 8);
    CHECK(a.doviConf->level == 6);
    CHECK(a.doviProfile == 8);
    CHECK(a.cls == SourceClass::Hlg);
    CHECK(a.refuseAll == ColourRefusal::None);
    REQUIRE(a.Note("source.dv_removed") != nullptr);
    CHECK_THAT(a.Note("source.dv_removed")->detail, ContainsSubstring("HLG base"));

    const auto ipt = clip("p5.mkv", RawTags{2, 2, 2, 2, 0}, 5, 0);
    VideoDecoder p5(ipt, VideoDecoder::Probe());
    const SourceColor& b = p5.Info().color;
    REQUIRE(b.doviConf);
    CHECK(b.doviConf->profile == 5);
    CHECK(b.refuseAll == ColourRefusal::DolbyVisionIpt);  // TB-1 marks it; TB-2 refuses
    CHECK(b.Note("source.dv_ipt") != nullptr);
    CHECK_FALSE(b.legacyPath);
}

TEST_CASE("dlssvid info prints colour mastering light level and Dolby Vision; an SDR clip prints as before", "[integration][hdr][cli]") {
    const auto dir = Dir("info");
    const auto hlg = RunAll({"info", ProResHlg(dir).string()}, dir / "hlg.log");
    {
        INFO(hlg.out);
        CHECK(hlg.code == 0);
        CHECK_THAT(hlg.out, ContainsSubstring("colour:      HDR HLG - matrix bt2020nc (stream), primaries bt2020 (stream), transfer arib-std-b67 "
                                              "(frame; stream bt2020-10)"));
        CHECK_THAT(hlg.out, ContainsSubstring("mastering:   R 0.708,0.292 G 0.17,0.797 B 0.131,0.046 white 0.3127,0.329, luminance 0.0001-1000 "
                                              "cd/m2 (container)"));
        CHECK_THAT(hlg.out, ContainsSubstring("light level: MaxCLL 1000, MaxFALL 400 cd/m2 (container)"));
        CHECK_THAT(hlg.out, ContainsSubstring("ambient:     illuminance 314 lux"));
    }
    ClipHdr light;
    light.light = Cll{1000, 400};
    const auto pq = RunAll({"info", FfvPq(dir, "pq.mkv", light).string()}, dir / "pq.log");
    {
        INFO(pq.out);
        CHECK(pq.code == 0);
        CHECK_THAT(pq.out, ContainsSubstring("colour:      HDR PQ - matrix bt2020nc (stream), primaries bt2020 (stream), transfer smpte2084 (stream)"));
        CHECK_THAT(pq.out, !ContainsSubstring("mastering:"));
    }
    ClipSpec dv;
    dv.width = 64;
    dv.height = 48;
    dv.frames = 2;
    DoviConf d;
    d.profile = 5;
    d.level = 6;
    dv.hdr.dovi = d;
    const auto ipt = RunAll({"info", WriteClip(dir / "dv5.mkv", dv).string()}, dir / "dv5.log");
    {
        INFO(ipt.out);
        CHECK(ipt.code == 0);
        CHECK_THAT(ipt.out, ContainsSubstring("dolby vision: profile 5, level 6, compatibility 0 (record): IPT base - colours come out wrong without the RPU"));
        CHECK_THAT(ipt.out, !ContainsSubstring("not supported"));
    }
    ClipSpec plain;
    plain.width = 64;
    plain.height = 48;
    plain.frames = 2;
    const auto sdr = RunAll({"info", WriteClip(dir / "sdr.mkv", plain).string()}, dir / "sdr.log");
    {
        INFO(sdr.out);
        CHECK(sdr.code == 0);
        CHECK_THAT(sdr.out, ContainsSubstring("video:       ffv1 64x48 yuv420p (bt601, limited range)"));
        CHECK_THAT(sdr.out, !ContainsSubstring("colour:"));
        CHECK_THAT(sdr.out, !ContainsSubstring("note:"));
    }
}

TEST_CASE("PQ FFV1 MKV with HDR10+ in packet side data says HDR10+ without a decode", "[integration][hdr]") {
    const auto dir = Dir("hdr10plus");
    const auto clip = FfvPq(dir, "pq_hdr10plus.mkv", {}, [](int64_t i) {
        std::vector<std::pair<int, std::string>> sd;
        sd.emplace_back(AV_PKT_DATA_DYNAMIC_HDR10_PLUS, Hdr10PlusSideData());
        (void)i;
        return sd;
    });
    const int64_t decodes = FirstFrameDecodes(), peeks = FirstPacketPeeks();
    VideoDecoder dec(clip, VideoDecoder::Probe());
    const SourceColor& c = dec.Info().color;
    CHECK(FirstFrameDecodes() == decodes);
    CHECK(FirstPacketPeeks() == peeks + 1);
    CHECK(c.hdr10Plus);
    CHECK(c.hdr10PlusFrom == TagFrom::Packet);
    CHECK(c.Note("source.hdr10plus") != nullptr);
    const auto info = RunAll({"info", clip.string()}, dir / "info.log");
    INFO(info.out);
    CHECK_THAT(info.out, ContainsSubstring("hdr10+:      present (first packet): not carried into results"));
    // without it: nothing said
    const auto plain = FfvPq(dir, "pq.mkv");
    VideoDecoder none(plain, VideoDecoder::Probe());
    CHECK_FALSE(none.Info().color.hdr10Plus);
}

// ---- NVENC (HEVC / AV1): the bitstream's own colour ----------------------------------------------------------------------------

namespace {

std::optional<std::filesystem::path> NvencClip(const std::filesystem::path& file, ClipSpec spec, std::string* why) {
    spec.width = 320;
    spec.height = 192;
    if (spec.frames == 24) spec.frames = 8;
    if (spec.pixFmt == "yuv420p") spec.pixFmt = "p010le";
    return WriteClipIfSupported(file, spec, why);
}

}  // namespace

#define REQUIRE_NVENC(var, file, spec)                                    \
    if (!HaveNvidia()) SKIP("no NVIDIA GPU");                             \
    std::string var##Why;                                                 \
    const auto var##Opt = NvencClip(file, spec, &var##Why);               \
    if (!var##Opt) SKIP("NVENC does not write the clip here: " << var##Why); \
    const std::filesystem::path var = *var##Opt

TEST_CASE("HEVC MP4 with VUI trc 14 and an alternative transfer SEI 18 is HLG from the frame", "[integration][hdr][gpu]") {
    const auto dir = Dir("hevc_sei");
    ClipSpec spec;
    spec.codec = "hevc_nvenc";
    spec.color = Tags(9, 9, 14);
    spec.videoPacket = [](int64_t, const std::string& au) { return WithNalBeforeSlice(au, AlternativeTransferSei(18)); };
    REQUIRE_NVENC(clip, dir / "hlg_sei.mp4", spec);
    // FFmpeg 7.1.1: the codec parameters keep the VUI's 14 (the SEI is in a picture), the frame says 18 (hevcdec: export_stream_params_from_sei)
    REQUIRE(Summarize(clip).colorTrc == 14);
    REQUIRE(FrameColors(clip).at(0).trc == 18);
    VideoDecoder dec(clip, VideoDecoder::Probe());
    const SourceColor& c = dec.Info().color;
    CHECK(c.looked == FirstLook::Decode);
    CHECK(c.cls == SourceClass::Hlg);
    CHECK(c.trcFrom == TagFrom::Frame);
    CHECK(c.stream.trc == 14);
    CHECK(c.notes.empty());
}

TEST_CASE("HEVC MP4 with colr 9/16/9 and no colour description in the VUI is PQ from the container", "[integration][hdr][gpu]") {
    const auto dir = Dir("hevc_colr");
    ClipSpec spec;
    spec.codec = "hevc_nvenc";
    spec.containerColor = Tags(9, 9, 16);
    REQUIRE_NVENC(clip, dir / "pq_colr.mp4", spec);
    // the demuxer read colr; avformat_find_stream_info put the VUI's unspecified over it
    const StreamSummary s = Summarize(clip);
    REQUIRE(s.container.trc == 16);
    REQUIRE(s.colorTrc == 2);
    VideoDecoder dec(clip, VideoDecoder::Probe());
    const SourceColor& c = dec.Info().color;
    CHECK(c.cls == SourceClass::Pq);
    CHECK(c.trcFrom == TagFrom::Container);
    CHECK(c.primariesFrom == TagFrom::Container);
    CHECK(c.matrixFrom == TagFrom::Container);
    CHECK(c.looked == FirstLook::Decode);
    CHECK(c.Note("source.layers_differ") == nullptr);  // nothing said against it
    CHECK(dec.Info().colorTrc == 2);  // the legacy fields as before
}

TEST_CASE("Raw HEVC is probed once and still reads from frame 0", "[integration][hdr][gpu][cache]") {
    const auto dir = Dir("hevc_raw");
    ClipSpec spec;
    spec.codec = "hevc_nvenc";
    spec.color = Tags(9, 9, 16);
    spec.frames = 6;
    REQUIRE_NVENC(clip, dir / "pq.hevc", spec);
    SetOld(clip, std::chrono::minutes(60));
    const int64_t start = FirstFrameDecodes();
    { VideoDecoder probe(clip, VideoDecoder::Probe()); CHECK(probe.Info().color.cls == SourceClass::Pq); }
    VideoDecoder dec(clip);
    CHECK(FirstFrameDecodes() == start + 1);
    CHECK(dec.Info().color.looked == FirstLook::Decode);
    CpuFrame f;
    int frames = 0;
    while (dec.NextFrame(f)) {
        if (frames == 0) CHECK(f.index == 0);
        ++frames;
    }
    CHECK(frames == 6);
}

TEST_CASE("HEVC MP4 with a dvvC record under strict unofficial reads back as Dolby Vision 8.4", "[integration][hdr][dovi][gpu]") {
    const auto dir = Dir("hevc_dovi");
    ClipSpec spec;
    spec.codec = "hevc_nvenc";
    spec.color = Tags(9, 9, 18);
    DoviConf d;
    d.major = 1;
    d.profile = 8;
    d.level = 4;
    d.blCompat = 4;
    d.rpu = d.bl = true;
    spec.hdr.dovi = d;
    spec.strict = FF_COMPLIANCE_UNOFFICIAL;
    REQUIRE_NVENC(clip, dir / "p84.mp4", spec);
    REQUIRE(Summarize(clip).dovi);
    VideoDecoder dec(clip, VideoDecoder::Probe());
    const SourceColor& c = dec.Info().color;
    REQUIRE(c.doviConf);
    CHECK(c.doviConf->profile == 8);
    CHECK(c.doviConf->blCompat == 4);
    CHECK(c.cls == SourceClass::Hlg);
    CHECK(c.Note("source.dv_removed") != nullptr);
}

TEST_CASE("AV1 colour comes from the codec parameters without a decode", "[integration][hdr][gpu]") {
    const auto dir = Dir("av1");
    ClipSpec spec;
    spec.codec = "av1_nvenc";
    spec.color = Tags(9, 9, 16);
    spec.gop = 8;
    REQUIRE_NVENC(clip, dir / "pq.ivf", spec);  // IVF: no colour of the container's own - the sequence header's color_config alone
    const int64_t decodes = FirstFrameDecodes(), peeks = FirstPacketPeeks();
    VideoDecoder dec(clip, VideoDecoder::Probe());
    const SourceColor& c = dec.Info().color;
    CHECK(FirstFrameDecodes() == decodes);
    CHECK(FirstPacketPeeks() == peeks);
    CHECK(c.container.trc == 2);
    CHECK(c.stream.trc == 16);
    CHECK(c.cls == SourceClass::Pq);
    REQUIRE(c.Note("source.av1_stream_metadata") != nullptr);
    CHECK_THAT(c.Note("source.av1_stream_metadata")->detail, ContainsSubstring("the container has none"));
}

// ---- review round 1 (TASK-0078) -------------------------------------------------------------------------------------------------

TEST_CASE("ProRes colour from the frame header: no colr box, and a header's 0 as unknown", "[integration][hdr]") {
    const auto dir = Dir("prores_header");
    // no colr (the encoder's tags unspecified: movenc writes none), the frame header says PQ of BT.2020 with no matrix
    ClipSpec spec;
    spec.width = 64;
    spec.height = 48;
    spec.frames = 2;
    spec.codec = "prores_ks";
    spec.pixFmt = "yuv422p10le";
    spec.frameColor = [](int64_t) { return RawTags{2, 9, 16, 1, 0}; };
    const auto bare = WriteClip(dir / "no_colr.mov", spec);
    REQUIRE(Summarize(bare).container == RawTags{2, 2, 2, Summarize(bare).container.range, Summarize(bare).container.chromaLoc});
    REQUIRE(FrameColors(bare).at(0).trc == 16);
    VideoDecoder a(bare, VideoDecoder::Probe());
    CHECK(a.Info().color.looked == FirstLook::Decode);
    CHECK(a.Info().color.cls == SourceClass::Pq);
    CHECK(a.Info().color.trcFrom == TagFrom::Frame);
    CHECK(a.Info().color.Note("source.bt2020_assumed") != nullptr);
    // colr 9/9/14, the header's curve 0 (FFmpeg's own prores encoder writes 0 for bt2020-10): the colr box's curve
    spec.color = Tags(9, 9, 14);
    spec.frameColor = [](int64_t) { return RawTags{9, 9, 0, 1, 0}; };
    const auto zero = WriteClip(dir / "trc0.mov", spec);
    REQUIRE(FrameColors(zero).at(0).trc == 0);
    VideoDecoder b(zero, VideoDecoder::Probe());
    CHECK(b.Info().color.cls == SourceClass::Wide);
    CHECK(b.Info().color.trcFrom == TagFrom::Stream);
    CHECK(b.Info().color.notes.empty());
}

TEST_CASE("A first look that fails is said, printed and cached", "[integration][hdr][cache]") {
    const auto dir = Dir("look_fails");
    ClipSpec spec;
    spec.width = 64;
    spec.height = 48;
    spec.frames = 2;
    spec.codec = "prores_ks";
    spec.pixFmt = "yuv422p10le";
    spec.color = Tags(9, 9, 14);
    spec.videoPacket = [](int64_t, const std::string& p) { return p.substr(0, 8); };  // no frame decodes from these
    const auto clip = WriteClip(dir / "broken.mov", spec);
    SetOld(clip, std::chrono::minutes(60));
    const int64_t start = FirstFrameDecodes();
    {
        VideoDecoder dec(clip, VideoDecoder::Probe());
        const SourceColor& c = dec.Info().color;
        CHECK_FALSE(c.probeError.empty());
        CHECK(c.lookFailed == FirstLook::Decode);
        CHECK(c.looked == FirstLook::None);
        CHECK(c.cls == SourceClass::Wide);  // from the colr box
        REQUIRE(c.Note("source.probe_failed") != nullptr);
    }
    { VideoDecoder again(clip, VideoDecoder::Probe()); CHECK_FALSE(again.Info().color.probeError.empty()); }
    CHECK(FirstFrameDecodes() == start + 1);  // the failure is an answer too
    const auto info = RunAll({"info", clip.string()}, dir / "info.log");
    INFO(info.out);
    CHECK_THAT(info.out, ContainsSubstring("note:        the first frame could not be read ("));
}

TEST_CASE("The first look's cache: a time ahead of the clock, the first packet, the least recently used file", "[integration][hdr][cache]") {
    const auto dir = Dir("cache_more");
    // a modification time an hour ahead (a camera card's local time): looked at once
    const auto ahead = ProResHlg(dir);
    std::filesystem::last_write_time(ahead, std::filesystem::file_time_type::clock::now() + std::chrono::hours(1));
    const int64_t decodes = FirstFrameDecodes();
    { VideoDecoder a(ahead, VideoDecoder::Probe()); }
    { VideoDecoder b(ahead, VideoDecoder::Probe()); }
    CHECK(FirstFrameDecodes() == decodes + 1);
    // the first packet of a PQ Matroska is read once; 64 other files later it is the least recently used and gone
    std::vector<std::filesystem::path> files;
    ClipSpec spec;
    spec.width = 64;
    spec.height = 48;
    spec.frames = 1;
    spec.pixFmt = "yuv420p10le";
    spec.color = Tags(9, 9, 16);
    for (int i = 0; i <= 64; ++i) {
        files.push_back(WriteClip(dir / ("pq" + std::to_string(i) + ".mkv"), spec));
        SetOld(files.back(), std::chrono::minutes(60));
    }
    const int64_t peeks = FirstPacketPeeks();
    { VideoDecoder a(files[0], VideoDecoder::Probe()); CHECK(a.Info().color.looked == FirstLook::Packet); }
    { VideoDecoder b(files[0], VideoDecoder::Probe()); CHECK(b.Info().color.looked == FirstLook::Packet); }
    CHECK(FirstPacketPeeks() == peeks + 1);
    for (int i = 1; i <= 64; ++i) VideoDecoder d(files[static_cast<size_t>(i)], VideoDecoder::Probe());
    CHECK(FirstPacketPeeks() == peeks + 65);
    { VideoDecoder last(files[64], VideoDecoder::Probe()); }
    CHECK(FirstPacketPeeks() == peeks + 65);  // the most recent ones stay
    { VideoDecoder first(files[0], VideoDecoder::Probe()); }
    CHECK(FirstPacketPeeks() == peeks + 66);
}
