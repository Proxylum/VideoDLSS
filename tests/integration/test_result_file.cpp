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

#include "TestClips.h"
#include "io/ResultFile.h"
#include "io/VideoDecoder.h"
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

TEST_CASE("CheckEncodedResult: the size, the frame count where the container keeps one, the audio", "[integration][result]") {
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
    const auto avi = WriteClip(dir / "c.avi", Small(4));  // AVI counts its frames
    CHECK(CheckEncodedResult(avi, {48, 32, 4, false}).empty());
    CHECK_THAT(CheckEncodedResult(avi, {48, 32, 5, false}), ContainsSubstring("4 frames") && ContainsSubstring("5 were encoded"));
    std::ofstream(dir / "junk.mkv", std::ios::binary) << "not a video";
    CHECK_THAT(CheckEncodedResult(dir / "junk.mkv", {48, 32, 4, false}), ContainsSubstring("does not open"));
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
    }
    SECTION("and a run that ends puts its result in place, nothing left beside it") {
        int64_t encodeTotal = -1;
        o.progress = [&](const std::string& stage, int64_t, int64_t total) {
            if (stage == "encode") encodeTotal = total;
        };
        const ProcessResult r = RunProcess(o);
        CHECK(r.warnings.empty());
        CHECK(DecodeAll(o.output).size() == 3);
        CHECK_FALSE(std::filesystem::exists(PartialResultPath(o.output)));
        CHECK(encodeTotal == 3);  // an MKV source counts no frames: the pass's - the GUI knows the encode step by its "N/M frames"
    }
}

TEST_CASE("RunProcess: a previous result open in a player is overwritten by a copy of the new one", "[integration][process][result][regression]") {
    // Windows renames nothing over an open file; FFmpeg's readers (the GUI's viewer, players) let others write, so a copy goes in
    const auto dir = Dir("open_in_player");
    const auto clip = WriteClip(dir / "clip.mkv", Small(3));
    ProcessOptions o = Options(dir, clip);
    WriteClip(o.output, Small(5));
    {
        VideoDecoder player(o.output);
        const ProcessResult r = RunProcess(o);
        CHECK(std::find(r.warnings.begin(), r.warnings.end(), std::string(kResultCopiedWarning)) != r.warnings.end());
    }
    CHECK(DecodeAll(o.output).size() == 3);
    CHECK_FALSE(std::filesystem::exists(PartialResultPath(o.output)));

    SECTION("not open: renamed, no warning") {
        CHECK(RunProcess(o).warnings.empty());
        CHECK(DecodeAll(o.output).size() == 3);
    }
}

TEST_CASE("PublishResult: a previous result that cannot be replaced stays; the new one stays beside it and the error says where", "[integration][result][regression]") {
    const auto dir = Dir("cannot_replace");
    const auto output = dir / "result.mkv";
    std::ofstream(output, std::ios::binary) << "the previous result";
    const auto partial = WriteClip(PartialResultPath(output), Small(3));
    {
        const Held held(output, FILE_SHARE_READ);  // no rename over it, no write into it
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
        CHECK_FALSE(PublishResult(partial, output, {48, 32, 3, false}).copied);
        CHECK(DecodeAll(output).size() == 3);
        CHECK_FALSE(std::filesystem::exists(partial));
    }
}
