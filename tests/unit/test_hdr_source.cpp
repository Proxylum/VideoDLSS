// TASK-0078 (HDR TB-1): what a source's colour is - the truth table, Dolby Vision, the layers, the first look's condition, the notes and
// the lines `dlssvid info` prints. Pure functions: no file, no decoder.
#include <catch2/catch_test_macros.hpp>

#include <string>
#include <vector>

#include "io/SourceColor.h"

using namespace dlssvid;

namespace {

RawTags Tags(int m, int p, int t, int range = 1) { return RawTags{m, p, t, range, 0}; }

// a source whose every layer says (m, p, t), classified
SourceColor Classified(int m, int p, int t, std::string_view layout = "420", std::string_view codec = "hevc") {
    SourceColor c;
    c.stream = c.effective = Tags(m, p, t);
    Classify(c, layout, codec);
    return c;
}

// the layers merged as the decoder merges them, then classified
SourceColor Layered(const RawTags& frame, const RawTags& stream, const RawTags& container, std::string_view codec = "hevc") {
    SourceColor c;
    c.frame = frame;
    c.stream = stream;
    c.container = container;
    TagFrom from[4];
    c.effective = MergeTags(frame, stream, container, from);
    c.matrixFrom = from[0];
    c.primariesFrom = from[1];
    c.trcFrom = from[2];
    c.rangeFrom = from[3];
    Classify(c, "420", codec);
    return c;
}

SourceColor WithDovi(int m, int p, int t, int profile, int compat) {
    SourceColor c;
    c.stream = c.effective = Tags(m, p, t);
    DoviConf d;
    d.major = 1;
    d.profile = profile;
    d.level = 6;
    d.blCompat = compat;
    d.rpu = d.bl = true;
    c.doviConf = d;
    Classify(c, "420", "hevc");
    return c;
}

SourceColor WithRpu(int m, int p, int t, int vdrProfile, bool fullRange, bool elResampling = false, bool disableResidual = false, int bitDepth = 10) {
    SourceColor c;
    c.stream = c.effective = Tags(m, p, t);
    DoviRpu r;
    r.vdrRpuProfile = vdrProfile;
    r.blFullRange = fullRange;
    r.elResampling = elResampling;
    r.disableResidual = disableResidual;
    r.vdrBitDepth = bitDepth;
    c.doviRpu = r;
    Classify(c, "420", "hevc");
    return c;
}

bool In(int v, std::initializer_list<int> set) {
    for (int s : set)
        if (v == s) return true;
    return false;
}

std::vector<std::string> Keys(const SourceColor& c) {
    std::vector<std::string> k;
    for (const ColourNote& n : c.notes) k.push_back(n.key);
    return k;
}

}  // namespace

TEST_CASE("HDR class follows the truth table rows over the whole M x P x T grid", "[hdr][color]") {
    int cells = 0;
    for (int m = 0; m <= 18; ++m)
        for (int p = 0; p <= 22; ++p)
            for (int t = 0; t <= 18; ++t) {
                const SourceColor c = Classified(m, p, t);
                // the design's rows, each as its own predicate (the design of TASK-0075, its truth table)
                const bool pq = t == 16, hlg = t == 18;
                const bool wideCurve = p == 9 && In(t, {1, 2, 6, 13, 14, 15});
                const bool log = p == 9 && !pq && !hlg && !wideCurve;
                const bool wideMatrix = m == 9 && p != 9 && !pq && !hlg;
                const SourceClass want = pq ? SourceClass::Pq
                                         : hlg ? SourceClass::Hlg
                                         : log ? SourceClass::Log
                                         : (wideCurve || wideMatrix) ? SourceClass::Wide
                                                                     : SourceClass::Legacy;
                INFO("m " << m << " p " << p << " t " << t);
                REQUIRE(c.cls == want);
                REQUIRE(c.rawCls == want);  // every layer says the same
                // what is refused: matrix 10 / 14 processing; matrix 15 (IPT) everything; an RGB matrix with HDR or BT.2020 everything
                REQUIRE((c.refuseProcessing == ColourRefusal::Matrix) == In(m, {10, 14}));
                const bool rgbHdr = m == 0 && (pq || hlg || p == 9);
                REQUIRE(c.refuseAll == (m == 15 ? ColourRefusal::DolbyVisionIpt : rgbHdr ? ColourRefusal::RgbHdr : ColourRefusal::None));
                REQUIRE(c.legacyPath == (want == SourceClass::Legacy && !In(m, {10, 14, 15}) && !rgbHdr));
                // the notes of the class's assumptions
                REQUIRE((c.Note("source.bt2020_assumed") != nullptr) == ((pq || hlg) && (m == 2 || p == 2)));
                REQUIRE((c.Note("source.trc_unspecified") != nullptr) == (p == 9 && t == 2));
                REQUIRE((c.Note("source.trc_not_converted") != nullptr) == log);
                REQUIRE((c.Note("source.matrix_2020") != nullptr) == wideMatrix);
                REQUIRE((c.Note("source.no_hdr_metadata") != nullptr) == pq);  // hevc, no records
                REQUIRE(c.Note("source.layers_differ") == nullptr);
                ++cells;
            }
    CHECK(cells == 19 * 23 * 19);
    // a few rows by name
    CHECK(Classified(1, 1, 1).legacyPath);
    CHECK(Classified(6, 6, 6).cls == SourceClass::Legacy);
    CHECK(Classified(2, 2, 2).legacyPath);
    CHECK(Classified(9, 9, 16).cls == SourceClass::Pq);
    CHECK(Classified(9, 9, 18).cls == SourceClass::Hlg);
    CHECK(Classified(9, 9, 14).cls == SourceClass::Wide);
    CHECK(Classified(9, 9, 8).cls == SourceClass::Log);
    CHECK(Classified(9, 1, 1).cls == SourceClass::Wide);
    CHECK(Classified(1, 1, 16).cls == SourceClass::Pq);  // PQ with BT.709 primaries: no BT.2020 assumed
    CHECK(Classified(1, 1, 16).Note("source.bt2020_assumed") == nullptr);
}

TEST_CASE("PQ and HLG with an untagged matrix or primaries assume BT.2020 and say so", "[hdr]") {
    for (int t : {16, 18}) {
        INFO("t " << t);
        CHECK(Classified(2, 9, t).Note("source.bt2020_assumed") != nullptr);
        CHECK(Classified(9, 2, t).Note("source.bt2020_assumed") != nullptr);
        CHECK(Classified(2, 2, t).Note("source.bt2020_assumed")->detail.find("matrix and primaries") != std::string::npos);
        CHECK(Classified(9, 9, t).Note("source.bt2020_assumed") == nullptr);
        CHECK(Classified(2, 9, t).Note("source.bt2020_assumed")->plan);
    }
}

TEST_CASE("Frame trc 18 over stream 1 6 14 or 15 is HLG without a note; other differences name both", "[hdr]") {
    for (int st : {1, 6, 14, 15}) {
        const SourceColor c = Layered(Tags(9, 9, 18), Tags(9, 9, st), RawTags{});
        INFO("stream trc " << st);
        CHECK(c.cls == SourceClass::Hlg);
        CHECK(c.trcFrom == TagFrom::Frame);
        CHECK(c.notes.empty());
    }
    // the frame says HLG over a PQ stream: another class from another layer - the plan hears it
    const SourceColor pqStream = Layered(Tags(9, 9, 18), Tags(9, 9, 16), RawTags{});
    CHECK(pqStream.cls == SourceClass::Hlg);
    CHECK(pqStream.rawCls == SourceClass::Pq);
    REQUIRE(pqStream.Note("source.layers_differ") != nullptr);
    CHECK(pqStream.Note("source.layers_differ")->plan);
    // the same class from two layers that differ: only `info` says it, with both values
    const SourceColor sameClass = Layered(Tags(6, 1, 1), Tags(1, 1, 1), RawTags{});
    CHECK(sameClass.cls == SourceClass::Legacy);
    CHECK(sameClass.legacyPath);
    REQUIRE(sameClass.Note("source.frame_differs") != nullptr);
    CHECK_FALSE(sameClass.Note("source.frame_differs")->plan);
    CHECK(sameClass.Note("source.frame_differs")->detail.find("smpte170m / bt709") != std::string::npos);
    // the container against the stream: of no consequence within a class (info), heard by the plan across classes
    const SourceColor sdrContainer = Layered(RawTags{}, Tags(6, 6, 6), Tags(1, 1, 1));
    REQUIRE(sdrContainer.Note("source.container_differs") != nullptr);
    CHECK_FALSE(sdrContainer.Note("source.container_differs")->plan);
    const SourceColor pqContainer = Layered(RawTags{}, Tags(1, 1, 1), Tags(9, 9, 16));
    CHECK(pqContainer.cls == SourceClass::Legacy);  // the stream's taken
    REQUIRE(pqContainer.Note("source.container_differs") != nullptr);
    CHECK(pqContainer.Note("source.container_differs")->plan);
}

TEST_CASE("RGB layout or matrix 0 with PQ HLG or BT.2020 primaries is refused in every mode", "[hdr]") {
    CHECK(Classified(1, 9, 16, "rgb").refuseAll == ColourRefusal::RgbHdr);
    CHECK(Classified(1, 1, 18, "rgb").refuseAll == ColourRefusal::RgbHdr);
    CHECK(Classified(1, 9, 1, "rgb").refuseAll == ColourRefusal::RgbHdr);
    CHECK(Classified(0, 9, 14, "420").refuseAll == ColourRefusal::RgbHdr);
    CHECK(Classified(0, 1, 16, "444").refuseAll == ColourRefusal::RgbHdr);
    CHECK(Classified(0, 9, 14, "420").Note("source.rgb_hdr") != nullptr);
    // RGB SDR is read as before
    CHECK(Classified(0, 1, 13, "rgb").refuseAll == ColourRefusal::None);
    CHECK(Classified(0, 1, 13, "rgb").legacyPath);
    CHECK(Classified(2, 2, 2, "rgb").legacyPath);
}

TEST_CASE("Dolby Vision verdict over profile and compatibility, reserved compatibilities and unknown profiles included", "[hdr][dovi]") {
    // an IPT base: profiles 1, 3, 5 and 10.0 - whatever the tags
    for (auto [p, cmp] : std::vector<std::pair<int, int>>{{5, 0}, {10, 0}, {1, 0}, {3, 0}, {5, 2}}) {
        INFO("profile " << p << " compatibility " << cmp);
        const SourceColor c = WithDovi(9, 9, 16, p, cmp);
        CHECK(c.refuseAll == ColourRefusal::DolbyVisionIpt);
        CHECK(c.Note("source.dv_ipt") != nullptr);
        CHECK_FALSE(c.legacyPath);
    }
    // a named base: HDR10 (1, 6), SDR (2), HLG (4) - the class from the tags, the RPU dropped
    const SourceColor p81 = WithDovi(9, 9, 16, 8, 1), p101 = WithDovi(9, 9, 16, 10, 1), p7 = WithDovi(9, 9, 16, 7, 6);
    const SourceColor p82 = WithDovi(1, 1, 1, 8, 2), p84 = WithDovi(9, 9, 18, 8, 4), p104 = WithDovi(9, 9, 18, 10, 4);
    for (const SourceColor* c : {&p81, &p101, &p7, &p82, &p84, &p104}) {
        CHECK(c->refuseAll == ColourRefusal::None);
        REQUIRE(c->Note("source.dv_removed") != nullptr);
        CHECK(c->Note("source.dv_removed")->plan);
    }
    CHECK(p81.cls == SourceClass::Pq);
    CHECK(p84.cls == SourceClass::Hlg);
    CHECK(p84.Note("source.dv_removed")->detail.find("HLG base") != std::string::npos);
    CHECK(p82.cls == SourceClass::Legacy);
    CHECK(p82.legacyPath);  // an SDR base is read as before - with its note
    CHECK(p7.Note("source.dv_removed")->detail.find("HDR10 base") != std::string::npos);
    // a record too short to say its base (FFmpeg: compatibility 0) of a profile that has one
    for (int p : {4, 7, 8, 9}) {
        const SourceColor c = WithDovi(9, 9, 16, p, 0);
        INFO("profile " << p);
        CHECK(c.refuseAll == ColourRefusal::None);
        CHECK(c.Note("source.dv_record_short") != nullptr);
        CHECK(c.cls == SourceClass::Pq);
    }
    // compatibilities FFmpeg does not name (3, 5, 7-15) and profiles without a base (0, 2, 6, 11+ at 0): untagged -> IPT, else the tags
    for (auto [p, cmp] : std::vector<std::pair<int, int>>{{8, 3}, {8, 5}, {8, 7}, {8, 15}, {11, 0}, {0, 0}, {2, 0}, {6, 0}, {12, 1}}) {
        INFO("profile " << p << " compatibility " << cmp);
        if (p >= 11 && cmp == 1) {  // a future profile with a named base: as any named base
            CHECK(WithDovi(9, 9, 16, p, cmp).Note("source.dv_removed") != nullptr);
            continue;
        }
        const SourceColor untagged = WithDovi(2, 2, 2, p, cmp);
        CHECK(untagged.refuseAll == ColourRefusal::DolbyVisionIpt);
        const SourceColor tagged = WithDovi(9, 9, 16, p, cmp);
        CHECK(tagged.refuseAll == ColourRefusal::None);
        CHECK(tagged.cls == SourceClass::Pq);
        CHECK(tagged.Note("source.dv_unknown") != nullptr);
        // only the matrix and the transfer decide it: untagged primaries do not
        CHECK(WithDovi(2, 9, 2, p, cmp).refuseAll == ColourRefusal::DolbyVisionIpt);
        CHECK(WithDovi(9, 2, 2, p, cmp).refuseAll == ColourRefusal::None);
    }
    // the RPU of the first frame, no record (a raw .hevc)
    CHECK(WithRpu(9, 9, 16, 0, true).refuseAll == ColourRefusal::DolbyVisionIpt);  // guessed 5
    CHECK(WithRpu(9, 9, 16, 1, false).Note("source.dv_removed") != nullptr);        // guessed 8
    CHECK(WithRpu(9, 9, 16, 1, false).doviProfile == 8);
    CHECK(WithRpu(2, 2, 2, 0, false).refuseAll == ColourRefusal::DolbyVisionIpt);   // unknown, untagged
    CHECK(WithRpu(9, 9, 16, 0, false).Note("source.dv_unknown") != nullptr);        // unknown, tagged
    // matrix 15 (IPT-C2) without any record
    CHECK(Classified(15, 9, 16).refuseAll == ColourRefusal::DolbyVisionIpt);
    CHECK(Classified(15, 9, 16).Note("source.dv_ipt") != nullptr);
}

TEST_CASE("Dolby Vision RPU guess follows FFmpeg 7.1.1", "[hdr][dovi]") {
    auto guess = [](int profile, bool full, bool el, bool noResidual, int depth) {
        DoviRpu r;
        r.vdrRpuProfile = profile;
        r.blFullRange = full;
        r.elResampling = el;
        r.disableResidual = noResidual;
        r.vdrBitDepth = depth;
        return GuessDoviProfile(r);
    };
    CHECK(guess(0, true, false, false, 12) == 5);
    CHECK(guess(0, false, false, false, 12) == 0);
    CHECK(guess(1, false, true, false, 12) == 7);
    CHECK(guess(1, false, true, false, 10) == 4);
    CHECK(guess(1, false, true, true, 12) == 8);
    CHECK(guess(1, false, false, false, 12) == 8);
    CHECK(guess(2, true, true, false, 12) == 0);
    CHECK(guess(-1, true, true, false, 12) == 0);
}

TEST_CASE("First look: SDR AV1 VP9 FFV1 none; HEVC trc 14 15, matrix 9, untagged 10-bit, raw streams, ProRes decode; PQ in Matroska reads a packet",
          "[hdr]") {
    const std::string mp4 = "mov,mp4,m4a,3gp,3g2,mj2", mkv = "matroska,webm";
    CHECK(NeedsFirstLook("hevc", mp4, 8, Tags(1, 1, 1)) == FirstLook::None);
    CHECK(NeedsFirstLook("h264", mp4, 8, Tags(6, 6, 6)) == FirstLook::None);
    CHECK(NeedsFirstLook("hevc", mp4, 8, Tags(2, 2, 2)) == FirstLook::None);  // 8-bit untagged: SDR
    CHECK(NeedsFirstLook("hevc", mp4, 10, Tags(2, 2, 2)) == FirstLook::Decode);
    CHECK(NeedsFirstLook("hevc", mp4, 10, Tags(1, 1, 1)) == FirstLook::None);
    for (int t : {14, 15, 16, 18}) {
        CHECK(NeedsFirstLook("hevc", mp4, 10, Tags(9, 9, t)) == FirstLook::Decode);
        CHECK(NeedsFirstLook("hevc", mp4, 8, Tags(1, 1, t)) == FirstLook::Decode);  // the curve alone is the hint
        CHECK(NeedsFirstLook("h264", mp4, 8, Tags(2, 2, t)) == FirstLook::Decode);
    }
    for (int m : {9, 10, 14, 15}) CHECK(NeedsFirstLook("h264", mp4, 8, Tags(m, 1, 1)) == FirstLook::Decode);
    CHECK(NeedsFirstLook("h264", mp4, 8, Tags(1, 9, 1)) == FirstLook::Decode);
    CHECK(NeedsFirstLook("hevc", "hevc", 8, Tags(1, 1, 1)) == FirstLook::Decode);  // a raw stream: no container to say it
    CHECK(NeedsFirstLook("h264", "h264", 8, Tags(1, 1, 1)) == FirstLook::Decode);
    CHECK(NeedsFirstLook("prores", mp4, 10, Tags(9, 9, 14)) == FirstLook::Decode);
    CHECK(NeedsFirstLook("prores", mp4, 10, Tags(1, 1, 1)) == FirstLook::None);
    // the codecs without a frame's own colour: nothing to decode
    for (const char* codec : {"av1", "vp9", "ffv1"}) {
        INFO(codec);
        CHECK(NeedsFirstLook(codec, mp4, 10, Tags(9, 9, 16)) == FirstLook::None);
        CHECK(NeedsFirstLook(codec, mkv, 10, Tags(9, 9, 18)) == FirstLook::None);
        CHECK(NeedsFirstLook(codec, mkv, 10, Tags(9, 9, 16)) == FirstLook::Packet);  // HDR10+ rides in a BlockAdditional
        CHECK(NeedsFirstLook(codec, mkv, 10, Tags(1, 1, 1)) == FirstLook::None);
    }
    CHECK(NeedsFirstLook("hevc", mkv, 10, Tags(9, 9, 16)) == FirstLook::Decode);  // a decode reads the packet's side data too
}

TEST_CASE("MergeTags takes frame over stream over container; a frame equal to the stream is the stream's; frame range is ignored", "[hdr]") {
    TagFrom from[4];
    // the frame differs: the frame's
    RawTags e = MergeTags(Tags(9, 9, 18), Tags(9, 9, 14), Tags(1, 1, 1), from);
    CHECK(e.trc == 18);
    CHECK(from[2] == TagFrom::Frame);
    CHECK(from[0] == TagFrom::Stream);  // equal to the stream's: FFmpeg filled the frame from it
    CHECK(from[1] == TagFrom::Stream);
    // the frame silent: the stream's, else the container's
    e = MergeTags(RawTags{}, Tags(2, 9, 2, 0), Tags(9, 1, 16, 2), from);
    CHECK(e == RawTags{9, 9, 16, 2, 0});
    CHECK(from[0] == TagFrom::Container);
    CHECK(from[1] == TagFrom::Stream);
    CHECK(from[2] == TagFrom::Container);
    CHECK(from[3] == TagFrom::Container);
    // everything silent
    e = MergeTags(RawTags{}, RawTags{2, 2, 2, 0, 0}, RawTags{2, 2, 2, 0, 0}, from);
    CHECK(e == RawTags{2, 2, 2, 0, 0});
    CHECK(from[0] == TagFrom::None);
    // a ProRes frame says limited whatever the stream: the container's full range stands
    e = MergeTags(Tags(9, 9, 18, 1), Tags(9, 9, 18, 0), Tags(9, 9, 18, 2), from);
    CHECK(e.range == 2);
    CHECK(from[3] == TagFrom::Container);
    // a frame's range alone says nothing
    e = MergeTags(Tags(1, 1, 1, 2), Tags(1, 1, 1, 0), Tags(1, 1, 1, 0), from);
    CHECK(e.range == 0);
    CHECK(from[3] == TagFrom::None);
    // the stream's range over the container's
    e = MergeTags(RawTags{}, Tags(1, 1, 1, 1), Tags(1, 1, 1, 2), from);
    CHECK(e.range == 1);
    CHECK(from[3] == TagFrom::Stream);
    // optional out-parameter
    CHECK(MergeTags(Tags(1, 1, 1), RawTags{}, RawTags{}).matrix == 1);
}

TEST_CASE("An AV1 PQ or HLG source says its stream metadata is neither read nor carried; HEVC PQ without metadata says none was found", "[hdr]") {
    const SourceColor av1Pq = Classified(9, 9, 16, "420", "av1");
    REQUIRE(av1Pq.Note("source.av1_stream_metadata") != nullptr);
    CHECK(av1Pq.Note("source.av1_stream_metadata")->detail.find("the container has none") != std::string::npos);
    CHECK(av1Pq.Note("source.no_hdr_metadata") == nullptr);
    SourceColor av1Hlg;
    av1Hlg.stream = av1Hlg.effective = Tags(9, 9, 18);
    av1Hlg.hdr.mdcv = Mdcv{};
    Classify(av1Hlg, "420", "av1");
    REQUIRE(av1Hlg.Note("source.av1_stream_metadata") != nullptr);
    CHECK(av1Hlg.Note("source.av1_stream_metadata")->detail.find("from the container: mastering display") != std::string::npos);
    CHECK(Classified(1, 1, 1, "420", "av1").Note("source.av1_stream_metadata") == nullptr);
    // HEVC
    CHECK(Classified(9, 9, 16).Note("source.no_hdr_metadata") != nullptr);
    CHECK(Classified(9, 9, 18).Note("source.no_hdr_metadata") == nullptr);  // HLG needs none
    SourceColor withLight;
    withLight.stream = withLight.effective = Tags(9, 9, 16);
    withLight.hdr.cll = Cll{1000, 400};
    Classify(withLight, "420", "hevc");
    CHECK(withLight.Note("source.no_hdr_metadata") == nullptr);
}

TEST_CASE("A source whose frames are SDR over PQ codec parameters is not on the legacy path", "[hdr]") {
    // a ProRes MOV: colr PQ (the codec parameters), frame headers BT.709
    const SourceColor c = Layered(Tags(1, 1, 1), Tags(9, 9, 16), Tags(9, 9, 16), "prores");
    CHECK(c.cls == SourceClass::Legacy);
    CHECK(c.rawCls == SourceClass::Pq);
    CHECK_FALSE(c.legacyPath);
    REQUIRE(c.Note("source.layers_differ") != nullptr);
    CHECK(c.Note("source.layers_differ")->plan);
    // and the other way round: frames PQ, colr SDR
    const SourceColor up = Layered(Tags(9, 9, 16), Tags(1, 1, 1), Tags(1, 1, 1), "prores");
    CHECK(up.cls == SourceClass::Pq);
    CHECK(up.rawCls == SourceClass::Legacy);
    CHECK_FALSE(up.legacyPath);
    CHECK(up.Note("source.layers_differ") != nullptr);
    // codec parameters silent, the container says PQ (an HEVC MP4 whose VUI says nothing): PQ, not legacy - nothing to reconcile
    const SourceColor silent = Layered(RawTags{}, RawTags{2, 2, 2, 1, 0}, Tags(9, 9, 16));
    CHECK(silent.cls == SourceClass::Pq);
    CHECK_FALSE(silent.legacyPath);
    CHECK(silent.Note("source.layers_differ") == nullptr);
}

TEST_CASE("HDR10+ and a failed first look are notes for the plan", "[hdr]") {
    SourceColor c;
    c.stream = c.effective = Tags(9, 9, 16);
    c.hdr10Plus = true;
    c.hdr10PlusFrom = TagFrom::Packet;
    c.probeError = "no frame in the first 64 video packets";
    Classify(c, "420", "ffv1");
    REQUIRE(c.Note("source.hdr10plus") != nullptr);
    CHECK(c.Note("source.hdr10plus")->detail.find("first packet") != std::string::npos);
    REQUIRE(c.Note("source.probe_failed") != nullptr);
    CHECK(c.Note("source.probe_failed")->detail.find("no frame in the first 64") != std::string::npos);
}

TEST_CASE("info colour lines: none for SDR, the class with each tag's layer, the records and Dolby Vision", "[hdr][cli]") {
    CHECK(ColourInfoLines(Classified(1, 1, 1)).empty());
    CHECK(ColourInfoLines(Layered(Tags(6, 1, 1), Tags(1, 1, 1), RawTags{})).empty());  // a difference of no consequence: SDR prints as before
    SourceColor hlg = Layered(Tags(9, 9, 18), Tags(9, 9, 14), RawTags{}, "prores");
    hlg.hdr.cll = Cll{1000, 400};
    hlg.cllFrom = TagFrom::Container;
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
    m.maxLum = {1000, 1};
    m.hasPrimaries = m.hasLuminance = true;
    hlg.hdr.mdcv = m;
    hlg.mdcvFrom = TagFrom::Container;
    DoviConf d;
    d.profile = 8;
    d.level = 6;
    d.blCompat = 4;
    hlg.doviConf = d;
    Classify(hlg, "422", "prores");
    const std::vector<std::string> lines = ColourInfoLines(hlg);
    REQUIRE(lines.size() >= 5);
    CHECK(lines[0] ==
          "colour:      HDR HLG - matrix bt2020nc (stream), primaries bt2020 (stream), transfer arib-std-b67 (frame; stream bt2020-10), range tv (stream)");
    CHECK(lines[1] == "mastering:   R 0.708,0.292 G 0.17,0.797 B 0.131,0.046 white 0.3127,0.329, luminance 0.0001-1000 cd/m2 (container)");
    CHECK(lines[2] == "light level: MaxCLL 1000, MaxFALL 400 cd/m2 (container)");
    CHECK(lines[3] == "dolby vision: profile 8, level 6, compatibility 4 (record): removed in every output");
    CHECK(lines[4].rfind("note:        Dolby Vision profile 8", 0) == 0);
    // an IPT base: the colours, not a refusal (TB-1 only says it)
    const std::vector<std::string> ipt = ColourInfoLines(WithDovi(9, 9, 16, 5, 0));
    bool said = false;
    for (const std::string& l : ipt) said = said || l == "dolby vision: profile 5, level 6, compatibility 0 (record): IPT base - colours come out wrong without the RPU";
    CHECK(said);
    for (const std::string& l : ipt) CHECK(l.find("not supported") == std::string::npos);
    // an SDR base: printed (the RPU is dropped), the class stays SDR
    const std::vector<std::string> sdrBase = ColourInfoLines(WithDovi(1, 1, 1, 8, 2));
    REQUIRE_FALSE(sdrBase.empty());
    CHECK(sdrBase[0].rfind("colour:      SDR - ", 0) == 0);
}

// ---- review round 1 (TASK-0078) -------------------------------------------------------------------------------------------------

namespace {
Mdcv Bt2020Display(bool reduced) {  // the same display as MP4 / an SEI writes it (1/50000, 1/10000) and as Matroska's floats read back
    Mdcv m;
    if (reduced) {
        m.primaries[0][0] = {177, 250};
        m.primaries[0][1] = {73, 250};
        m.primaries[1][0] = {17, 100};
        m.primaries[1][1] = {797, 1000};
        m.primaries[2][0] = {131, 1000};
        m.primaries[2][1] = {23, 500};
        m.white[0] = {3127, 10000};
        m.white[1] = {329, 1000};
        m.minLum = {1, 10000};
        m.maxLum = {1000, 1};
    } else {
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
    }
    m.hasPrimaries = m.hasLuminance = true;
    return m;
}

FirstLookColor Decoded(RawTags tags) {
    FirstLookColor f;
    f.how = FirstLook::Decode;
    f.tags = tags;
    return f;
}
}  // namespace

TEST_CASE("Reserved tag values are unspecified; a ProRes header's 0 is unknown, not RGB", "[hdr]") {
    CHECK(WithoutReserved(RawTags{0, 0, 0, 1, 0}, "prores") == RawTags{2, 2, 2, 1, 0});
    CHECK(WithoutReserved(RawTags{0, 0, 0, 1, 0}, "hevc") == RawTags{0, 2, 2, 1, 0});  // matrix 0 is RGB (GBR) outside ProRes
    CHECK(WithoutReserved(RawTags{3, 3, 3, 2, 1}, "hevc") == RawTags{2, 2, 2, 2, 1});
    CHECK(WithoutReserved(RawTags{9, 9, 16, 1, 0}, "prores") == RawTags{9, 9, 16, 1, 0});
    // FFmpeg's own prores encoder writes 0 for a curve outside its list (bt2020-10): the colr box's 14 stands
    ColourLayers in;
    in.container = in.stream = Tags(9, 9, 14);
    in.first = Decoded(RawTags{9, 9, 0, 1, 0});
    const SourceColor c = SourceColorOf(in, "422", "prores");
    CHECK(c.cls == SourceClass::Wide);
    CHECK(c.effective.trc == 14);
    CHECK(c.trcFrom == TagFrom::Stream);
    CHECK(c.notes.empty());
    // a matrix outside its list (bt2020c -> 0) under PQ: unknown, not an RGB refusal
    in.container = in.stream = RawTags{};
    in.first = Decoded(RawTags{0, 9, 16, 1, 0});
    const SourceColor pq = SourceColorOf(in, "422", "prores");
    CHECK(pq.cls == SourceClass::Pq);
    CHECK(pq.refuseAll == ColourRefusal::None);
    CHECK(pq.Note("source.bt2020_assumed") != nullptr);
}

TEST_CASE("HDR records are compared by value: Matroska's fractions and an SEI's are the same display", "[hdr]") {
    CHECK(SameRecord(Bt2020Display(true), Bt2020Display(false)));
    CHECK_FALSE(Bt2020Display(true) == Bt2020Display(false));  // other fractions
    Mdcv other = Bt2020Display(false);
    other.maxLum = {4000, 1};
    CHECK_FALSE(SameRecord(other, Bt2020Display(true)));
    Mdcv white = Bt2020Display(false);
    white.white[0] = {15700, 50000};
    CHECK_FALSE(SameRecord(white, Bt2020Display(false)));
    Mdcv lumOnly = Bt2020Display(false);
    lumOnly.hasPrimaries = false;
    CHECK_FALSE(SameRecord(lumOnly, Bt2020Display(false)));
    Mdcv lumOnly2 = Bt2020Display(true);
    lumOnly2.hasPrimaries = false;
    lumOnly2.primaries[0][0] = {0, 1};  // primaries not said: not compared
    CHECK(SameRecord(lumOnly, lumOnly2));
    CHECK(SameRecord(Amve{{314, 1}, {3127, 10000}, {329, 1000}}, Amve{{3140000, 10000}, {15635, 50000}, {16450, 50000}}));
    CHECK_FALSE(SameRecord(Amve{{314, 1}, {3127, 10000}, {329, 1000}}, Amve{{315, 1}, {3127, 10000}, {329, 1000}}));
}

TEST_CASE("SourceColorOf takes the container's records, the codec's where it has none, the frame's where they differ", "[hdr]") {
    ColourLayers in;
    in.container = in.stream = Tags(9, 9, 16);
    in.containerRecords.mdcv = Bt2020Display(true);  // a Matroska source
    in.containerRecords.cll = Cll{1000, 400};
    FirstLookColor first = Decoded(Tags(9, 9, 16));
    first.hdr.mdcv = Bt2020Display(false);  // the SEI's: the same display
    first.hdr.cll = Cll{1000, 400};
    in.first = first;
    SourceColor c = SourceColorOf(in, "420", "hevc");
    CHECK(c.mdcvFrom == TagFrom::Container);
    CHECK(c.cllFrom == TagFrom::Container);
    CHECK(c.hdr.mdcv == Bt2020Display(true));
    CHECK(c.looked == FirstLook::Decode);
    // the frame says another display and light: the frame's
    first.hdr.mdcv->maxLum = {40000000, 10000};
    first.hdr.cll = Cll{4000, 1000};
    first.hdr.amve = Amve{{314, 1}, {3127, 10000}, {329, 1000}};
    in.first = first;
    c = SourceColorOf(in, "420", "hevc");
    CHECK(c.mdcvFrom == TagFrom::Frame);
    CHECK(c.hdr.mdcv->maxLum.Value() == 4000.0);
    CHECK(c.cllFrom == TagFrom::Frame);
    CHECK(c.hdr.cll == Cll{4000, 1000});
    CHECK(c.amveFrom == TagFrom::Frame);  // none in the container
    // the codec's records (an hvcC's SEI) where the container has none
    ColourLayers codec;
    codec.stream = Tags(9, 9, 16);
    codec.codecRecords.mdcv = Bt2020Display(false);
    codec.codecRecords.cll = Cll{1000, 400};
    codec.containerRecords.cll = Cll{800, 300};
    c = SourceColorOf(codec, "420", "hevc");
    CHECK(c.mdcvFrom == TagFrom::Stream);
    CHECK(c.cllFrom == TagFrom::Container);
    CHECK(c.hdr.cll == Cll{800, 300});
    CHECK(c.Note("source.no_hdr_metadata") == nullptr);
    // the RPU and HDR10+ of a decoded frame
    FirstLookColor dv = Decoded(Tags(9, 9, 16));
    DoviRpu rpu;
    rpu.vdrRpuProfile = 1;
    dv.rpu = rpu;
    dv.hdr10Plus = true;
    codec.first = dv;
    c = SourceColorOf(codec, "420", "hevc");
    REQUIRE(c.doviRpu);
    CHECK(c.doviProfile == 8);
    CHECK(c.Note("source.dv_removed") != nullptr);
    CHECK(c.hdr10Plus);
    CHECK(c.hdr10PlusFrom == TagFrom::Frame);
    // a look that failed: said, with what it was
    FirstLookColor failed;
    failed.how = FirstLook::Packet;
    failed.error = "no video packet";
    codec.first = failed;
    c = SourceColorOf(codec, "420", "ffv1");
    CHECK(c.lookFailed == FirstLook::Packet);
    CHECK(c.looked == FirstLook::None);
    REQUIRE(c.Note("source.probe_failed") != nullptr);
    CHECK(c.Note("source.probe_failed")->detail.rfind("the first video packet could not be read (no video packet)", 0) == 0);
    failed.how = FirstLook::Decode;
    codec.first = failed;
    CHECK(SourceColorOf(codec, "420", "hevc").Note("source.probe_failed")->detail.rfind("the first frame could not be read", 0) == 0);
}

TEST_CASE("The container is compared with what was taken: HLG from an SEI is no news, a container's other class is", "[hdr]") {
    // colr 18, VUI 14, the SEI's 18 in the frame: the container agrees with what was taken
    const SourceColor hevc = Layered(Tags(9, 9, 18), Tags(9, 9, 14), Tags(9, 9, 18));
    CHECK(hevc.cls == SourceClass::Hlg);
    CHECK(hevc.notes.empty());
    // H.264: find_stream_info decodes, the SEI's 18 is in the codec parameters already; colr says the VUI's 14
    const SourceColor h264 = Layered(RawTags{}, Tags(9, 9, 18), Tags(9, 9, 14), "h264");
    CHECK(h264.cls == SourceClass::Hlg);
    CHECK(h264.notes.empty());
    // the container says PQ, the VUI nothing, the frame HLG: another class - the plan hears it
    const SourceColor pqOverHlg = Layered(Tags(9, 9, 18), RawTags{2, 2, 2, 1, 0}, Tags(9, 9, 16));
    CHECK(pqOverHlg.cls == SourceClass::Hlg);
    REQUIRE(pqOverHlg.Note("source.container_differs") != nullptr);
    CHECK(pqOverHlg.Note("source.container_differs")->plan);
    CHECK(pqOverHlg.Note("source.container_differs")->detail.find("transfer smpte2084 / arib-std-b67") != std::string::npos);
    const std::vector<TagDifference> d = ContainerDifferences(pqOverHlg);
    REQUIRE(d.size() == 1);
    CHECK(d[0].field == "transfer");
    CHECK(d[0].a == 16);
    CHECK(d[0].b == 18);
    // a matrix that differs beside the SEI's curve is news
    const SourceColor matrix = Layered(Tags(1, 9, 18), Tags(9, 9, 14), RawTags{});
    CHECK(matrix.Note("source.layers_differ") != nullptr);
}

TEST_CASE("A ProRes without all three tags in its container is looked at", "[hdr]") {
    const std::string mov = "mov,mp4,m4a,3gp,3g2,mj2";
    CHECK(NeedsFirstLook("prores", mov, 10, RawTags{2, 2, 2, 0, 0}) == FirstLook::Decode);
    CHECK(NeedsFirstLook("prores", mov, 10, Tags(1, 1, 2)) == FirstLook::Decode);
    CHECK(NeedsFirstLook("prores", mov, 10, Tags(2, 1, 1)) == FirstLook::Decode);
    CHECK(NeedsFirstLook("prores", mov, 10, Tags(1, 1, 1)) == FirstLook::None);
    CHECK(NeedsFirstLook("ffv1", mov, 10, RawTags{2, 2, 2, 0, 0}) == FirstLook::None);
}

TEST_CASE("TB-1 says what a source cannot get without refusing it: ICtCp, BT.2020 CL, RGB HDR", "[hdr][cli]") {
    for (const SourceColor& c : {Classified(14, 9, 16), Classified(10, 9, 14), Classified(0, 9, 16, "rgb"), Classified(1, 9, 1, "rgb")}) {
        const std::vector<std::string> lines = ColourInfoLines(c);
        REQUIRE_FALSE(lines.empty());
        for (const std::string& l : lines) {
            INFO(l);
            CHECK(l.find("not supported") == std::string::npos);
            CHECK(l.find("refused") == std::string::npos);
        }
    }
    CHECK(Classified(14, 9, 16).Note("source.matrix_unsupported")->detail == "matrix ictcp: read as BT.709 / BT.601 - its colours come out wrong when processed");
    CHECK(Classified(0, 9, 16, "rgb").Note("source.rgb_hdr")->detail ==
          "an RGB stream of smpte2084 / bt2020: read as 8-bit 4:2:0 SDR through swscale - its colours come out wrong");
}
