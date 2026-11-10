// TASK-0070: the result of `process` replaces the previous one only when it is complete and checked - a run cancelled or failed while
// it encodes leaves the previous result as it was.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <windows.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "TestClips.h"
#include "io/ResultFile.h"
#include "io/VideoDecoder.h"
#include "io/VideoEncoder.h"
#include "pipeline/ProcessRunner.h"

using namespace dlssvid;
using namespace dlssvid::test;
using Catch::Matchers::ContainsSubstring;

namespace {

std::filesystem::path Dir(const char* name) {
    const auto d = TempDir() / "result_file" / name;
    std::filesystem::remove_all(d);
    std::filesystem::create_directories(d);
    return d;
}

std::string Bytes(const std::filesystem::path& p) {
    std::ifstream f(p, std::ios::binary);
    return {std::istreambuf_iterator<char>(f), {}};
}

ClipSpec Small(int frames) {
    ClipSpec spec;
    spec.frames = frames;
    spec.width = 48;
    spec.height = 32;
    return spec;
}

// No fg: the result has the source's frames.
std::vector<ProcessStage> StubStages() {
    std::vector<ProcessStage> s = DefaultProcessStages();
    for (const char* spec : {"depth.backend=stub", "flow.backend=stub", "upscale.backend=nis", "upscale.scale=2", "nr.backend=stub"}) ApplyParamSpec(s, spec);
    s.erase(std::remove_if(s.begin(), s.end(), [](const ProcessStage& st) { return st.name == "fg"; }), s.end());
    return s;
}

ProcessOptions Options(const std::filesystem::path& dir, const std::filesystem::path& clip) {
    ProcessOptions o;
    o.input = clip;
    o.output = dir / u8"результат.mkv";
    o.passesRoot = dir / "passes";
    o.stages = StubStages();
    o.codec = "ffv1";
    o.warp = true;
    return o;
}

// A handle that lets others read only - what a program that opens a file for itself does. Windows then neither renames over it nor
// writes into it.
struct Held {
    HANDLE h;
    explicit Held(const std::filesystem::path& p, DWORD share) : h(CreateFileW(p.c_str(), GENERIC_READ, share, nullptr, OPEN_EXISTING, 0, nullptr)) {
        REQUIRE(h != INVALID_HANDLE_VALUE);
    }
    ~Held() { CloseHandle(h); }
};

}  // namespace

TEST_CASE("PartialResultPath: next to the result, its extension kept for the container, a non-ASCII name kept", "[integration][result]") {
    const std::filesystem::path out = std::filesystem::path("D:/v") / u8"клип_result.mp4";
    CHECK(PartialResultPath(out) == std::filesystem::path("D:/v") / u8"клип_result.dlssvid-part.mp4");
    CHECK(PartialResultPath(out).extension() == ".mp4");
    CHECK(PartialResultPath("r.mkv") == std::filesystem::path("r.dlssvid-part.mkv"));
}

TEST_CASE("CheckEncodedResult: the size, the frame count where the container counts its samples, the audio", "[integration][result]") {
    const auto dir = Dir("check");
    ClipSpec spec = Small(4);
    spec.audio = true;
    const auto mkv = WriteClip(dir / "a.mkv", spec);
    CHECK(CheckEncodedResult(mkv, {48, 32, 4, true}).empty());
    CHECK_THAT(CheckEncodedResult(mkv, {96, 32, 4, true}), ContainsSubstring("48x32") && ContainsSubstring("96x32"));
    CHECK(CheckEncodedResult(mkv, {48, 32, 7, true}).empty());  // MKV keeps no count: not checked
    const auto silent = WriteClip(dir / "b.mkv", Small(4));
    CHECK_THAT(CheckEncodedResult(silent, {48, 32, 4, true}), ContainsSubstring("no audio"));
    CHECK(CheckEncodedResult(silent, {48, 32, 4, false}).empty());
    // AVI's count takes in the empty frames its muxer pads time gaps with: not checked (review of TASK-0070)
    const auto avi = WriteClip(dir / "c.avi", Small(4));
    CHECK(CheckEncodedResult(avi, {48, 32, 5, false}).empty());
    std::ofstream(dir / "junk.mkv", std::ios::binary) << "not a video";
    CHECK_THAT(CheckEncodedResult(dir / "junk.mkv", {48, 32, 4, false}), ContainsSubstring("does not open"));
    if (!VideoEncoder::EncoderAvailable("prores_ks")) SKIP("prores_ks unavailable: MOV not checked");
    ClipSpec prores = Small(4);
    prores.codec = "prores_ks";
    prores.pixFmt = "yuv422p10le";
    const auto mov = WriteClip(dir / "d.mov", prores);  // MOV counts its samples, one per frame
    CHECK(CheckEncodedResult(mov, {48, 32, 4, false}).empty());
    CHECK_THAT(CheckEncodedResult(mov, {48, 32, 5, false}), ContainsSubstring("4 frames") && ContainsSubstring("5 were encoded"));
}

TEST_CASE("RunProcess: a run that fails while it encodes keeps the previous result and leaves no unfinished file", "[integration][process][result][regression]") {
    // TASK-0070: the encoder opened the result file itself and emptied it at once - a cancel (the GUI kills the CLI) or a failure at
    // the encode step destroyed the previous result
    const auto dir = Dir("fails_encoding");
    const auto clip = WriteClip(dir / "clip.mkv", Small(3));
    ProcessOptions o = Options(dir, clip);
    std::ofstream(o.output, std::ios::binary) << "the previous result";
    o.progress = [](const std::string& stage, int64_t done, int64_t) {
        if (stage == "encode" && done >= 2) throw std::runtime_error("stopped at the encode");  // after it has written frames
    };
    CHECK_THROWS_WITH(RunProcess(o), ContainsSubstring("stopped at the encode"));
    CHECK(Bytes(o.output) == "the previous result");
    CHECK_FALSE(std::filesystem::exists(PartialResultPath(o.output)));

    // what the GUI goes by: "encode: N/M frames" lines, the first one (frame 0) before the file is made - it lets the shown result go
    // then and removes the file a cancel leaves (review of TASK-0070: MKV sources gave no total, the first line came at frame 10)
    std::vector<std::pair<int64_t, int64_t>> encodeSteps;
    bool fileBeforeFirst = true;
    const auto track = [&](const std::string& stage, int64_t done, int64_t total) {
        if (stage != "encode") return;
        if (encodeSteps.empty()) fileBeforeFirst = std::filesystem::exists(PartialResultPath(o.output));
        encodeSteps.emplace_back(done, total);
    };
    SECTION("the same for a run without stages (the source passed through)") {
        o.passthrough = true;
        o.progress = [](const std::string& stage, int64_t done, int64_t) {
            if (stage == "passthrough" && done >= 2) throw std::runtime_error("stopped in the passthrough");
        };
        CHECK_THROWS_WITH(RunProcess(o), ContainsSubstring("stopped in the passthrough"));
        CHECK(Bytes(o.output) == "the previous result");
        CHECK_FALSE(std::filesystem::exists(PartialResultPath(o.output)));
    }
    SECTION("the same when no stage makes a colour pass: the source is encoded as it is, as the run's encode step") {
        o.stages.clear();
        o.passesRoot = dir / "no_passes";  // the run above left color_nr in "passes"
        CHECK_THROWS_WITH(RunProcess(o), ContainsSubstring("stopped at the encode"));
        CHECK(Bytes(o.output) == "the previous result");
        CHECK_FALSE(std::filesystem::exists(PartialResultPath(o.output)));
        o.progress = track;
        RunProcess(o);
        REQUIRE(encodeSteps.size() >= 2);
        CHECK(encodeSteps.front().first == 0);
        CHECK_FALSE(fileBeforeFirst);
        for (const auto& [done, total] : encodeSteps) CHECK((total >= 3 && done <= total));  // estimated from the MKV's duration
    }
    SECTION("and a run that ends puts its result in place, nothing left beside it") {
        o.progress = track;
        const ProcessResult r = RunProcess(o);
        CHECK(r.warnings.empty());
        CHECK(DecodeAll(o.output).size() == 3);
        CHECK_FALSE(std::filesystem::exists(PartialResultPath(o.output)));
        REQUIRE(encodeSteps.size() == 4);
        CHECK(encodeSteps.front() == std::pair<int64_t, int64_t>{0, 3});  // an MKV source counts no frames: the pass's
        CHECK(encodeSteps.back() == std::pair<int64_t, int64_t>{3, 3});
        CHECK_FALSE(fileBeforeFirst);
    }
}

TEST_CASE("RunProcess: a previous result open in a player stays; the new one stays beside it and the error says where", "[integration][process][result][regression]") {
    // Windows renames nothing over a file a reader without delete sharing has open - FFmpeg's readers, so most players; overwriting it in
    // place would not be atomic (review of TASK-0070: a cancel during such a copy lost both results)
    const auto dir = Dir("open_in_player");
    const auto clip = WriteClip(dir / "clip.mkv", Small(3));
    ProcessOptions o = Options(dir, clip);
    WriteClip(o.output, Small(5));
    const auto partial = PartialResultPath(o.output);
    {
        VideoDecoder player(o.output);
        CHECK_THROWS_WITH(RunProcess(o), ContainsSubstring(partial.string()) && ContainsSubstring("could not be replaced"));
    }
    CHECK(DecodeAll(o.output).size() == 5);
    CHECK(DecodeAll(partial).size() == 3);
    // closed: the run again puts it in place (the passes are reused)
    CHECK(RunProcess(o).warnings.empty());
    CHECK(DecodeAll(o.output).size() == 3);
    CHECK_FALSE(std::filesystem::exists(partial));
}

TEST_CASE("PublishResult: a previous result that cannot be replaced stays; the new one stays beside it and the error says where", "[integration][result][regression]") {
    const auto dir = Dir("cannot_replace");
    const auto output = dir / "result.mkv";
    std::ofstream(output, std::ios::binary) << "the previous result";
    const auto partial = WriteClip(PartialResultPath(output), Small(3));
    {
        const Held held(output, FILE_SHARE_READ | FILE_SHARE_WRITE);  // as FFmpeg opens it
        CHECK_THROWS_WITH(PublishResult(partial, output, {48, 32, 3, false}), ContainsSubstring(partial.string()) && ContainsSubstring("could not be replaced"));
    }
    CHECK(Bytes(output) == "the previous result");
    CHECK(DecodeAll(partial).size() == 3);

    SECTION("a new file that is not what was encoded does not replace it either") {
        CHECK_THROWS_WITH(PublishResult(partial, output, {64, 32, 3, false}), ContainsSubstring("is not right") && ContainsSubstring("64x32"));
        CHECK(Bytes(output) == "the previous result");
        CHECK(std::filesystem::exists(partial));  // left for a look
    }
    SECTION("free: it goes in place") {
        PublishResult(partial, output, {48, 32, 3, false});
        CHECK(DecodeAll(output).size() == 3);
        CHECK_FALSE(std::filesystem::exists(partial));
    }
}

namespace {
// A clip whose audio track has no packets: MKV keeps such a track, MP4 and MOV leave it out.
std::filesystem::path ClipWithEmptyAudio(const std::filesystem::path& dir) {
    ClipSpec donorSpec = Small(1);
    donorSpec.audio = true;
    VideoDecoder donor(WriteClip(dir / "donor.mkv", donorSpec));
    VideoEncoder::Options eo;
    eo.codec = "ffv1";
    eo.frameRate = Rational{24, 1};
    const auto path = dir / "empty_audio.mkv";
    VideoEncoder enc(path, FrameDesc{48, 32, PixelFormat::Yuv420p}, eo);
    enc.AddAudioStreamCopy(donor.AudioCodecParameters(), donor.AudioTimeBase());
    enc.Open();
    for (int64_t i = 0; i < 3; ++i) {
        CpuFrame f = SyntheticFrame(i, 48, 32);
        f.index = i;
        f.pts = i;
        enc.WriteFrame(f);
    }
    enc.Close();
    return path;
}
}  // namespace

TEST_CASE("RunProcess: a source with an empty audio track gives a MOV result without audio, not a rejected one", "[integration][process][result][regression]") {
    // review of TASK-0070: audio was expected whenever the source had a track, and MOV leaves a track without packets out
    if (!VideoEncoder::EncoderAvailable("prores_ks")) SKIP("prores_ks unavailable");
    const auto dir = Dir("empty_audio");
    const auto clip = ClipWithEmptyAudio(dir);
    REQUIRE(VideoDecoder(clip, VideoDecoder::Probe()).HasAudio());
    ProcessOptions o = Options(dir, clip);
    o.output = dir / "result.mov";
    o.codec = "prores_ks";
    SECTION("from the passes") {}
    SECTION("passed through") { o.passthrough = true; }
    const ProcessResult r = RunProcess(o);
    CHECK_FALSE(r.audio);
    CHECK(Summarize(o.output).audioStreams == 0);
    CHECK(DecodeAll(o.output).size() == 3);
}

TEST_CASE("RunProcess: a passthrough into AVI from an MKV source puts its result in place", "[integration][process][result][regression]") {
    // review of TASK-0070: the source's time base is kept, AVI pads the gaps with empty frames and counts them - the check rejected it
    const auto dir = Dir("avi");
    const auto clip = WriteClip(dir / "clip.mkv", Small(4));
    ProcessOptions o = Options(dir, clip);
    o.output = dir / "result.avi";
    o.passthrough = true;
    RunProcess(o);
    CHECK(DecodeAll(o.output).size() == 4);
    CHECK_FALSE(std::filesystem::exists(PartialResultPath(o.output)));
}

TEST_CASE("RunProcess: a read-only previous result is refused before anything runs; PublishResult names the cause", "[integration][process][result][regression]") {
    // review of TASK-0070: nothing renames over a read-only file - the run failed only after the whole encode, asking about a player
    const auto dir = Dir("read_only");
    const auto clip = WriteClip(dir / "clip.mkv", Small(3));
    ProcessOptions o = Options(dir, clip);
    std::ofstream(o.output, std::ios::binary) << "the previous result";
    constexpr auto kWrite = std::filesystem::perms::owner_write | std::filesystem::perms::group_write | std::filesystem::perms::others_write;
    std::filesystem::permissions(o.output, kWrite, std::filesystem::perm_options::remove);  // MSVC: the read-only attribute
    REQUIRE(ReadOnlyFile(o.output));
    CHECK_THROWS_WITH(RunProcess(o), ContainsSubstring("read-only"));
    CHECK_FALSE(std::filesystem::exists(o.passesRoot / "color_nr"));  // refused before the stages
    CHECK_FALSE(std::filesystem::exists(PartialResultPath(o.output)));

    const auto partial = WriteClip(PartialResultPath(o.output), Small(3));
    CHECK_THROWS_WITH(PublishResult(partial, o.output, {48, 32, 3, false}), ContainsSubstring("read-only") && !ContainsSubstring("player"));
    std::filesystem::permissions(o.output, kWrite, std::filesystem::perm_options::add);
    CHECK_FALSE(ReadOnlyFile(o.output));
    CHECK(Bytes(o.output) == "the previous result");
    CHECK(std::filesystem::exists(partial));
}
