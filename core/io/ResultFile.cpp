#include "io/ResultFile.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <system_error>
#include <thread>

#include "io/VideoDecoder.h"
#include "util/Error.h"
#include "util/Log.h"

namespace dlssvid {

namespace fs = std::filesystem;

namespace {
// An operation the file system refused for a moment: an antivirus opens a file that has just been written, the GUI's viewer lets the
// result go on the run's first encode line (TASK-0070). A file held for good still fails.
template <class Op>
void Retrying(std::error_code& ec, int attempts, Op op) {
    for (int attempt = 1;; ++attempt) {
        ec.clear();
        op(ec);
        if (!ec || attempt >= attempts) return;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
}

// MP4 and MOV count their samples: one per encoded frame. AVI's count includes the empty frames its muxer pads time gaps with (a
// passthrough keeps the source's time base); MKV keeps none.
bool CountsFrames(const fs::path& file) {
    std::string ext = file.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return ext == ".mp4" || ext == ".mov" || ext == ".m4v";
}
}  // namespace

fs::path PartialResultPath(const fs::path& output) {
    fs::path name = output.stem();
    name += ".dlssvid-part";
    name += output.extension();
    return output.parent_path() / name;
}

std::string CheckEncodedResult(const fs::path& file, const ResultExpectation& expect) {
    try {
        VideoDecoder probe(file, VideoDecoder::Probe());
        const VideoStreamInfo& info = probe.Info();
        if (info.width != expect.width || info.height != expect.height)
            return "the video is " + std::to_string(info.width) + "x" + std::to_string(info.height) + ", " + std::to_string(expect.width) + "x" +
                   std::to_string(expect.height) + " was encoded";
        if (expect.frames >= 0 && CountsFrames(file) && info.frameCount != expect.frames)
            return "it holds " + std::to_string(info.frameCount) + " frames, " + std::to_string(expect.frames) + " were encoded";
        if (expect.audio && !probe.HasAudio()) return "it has no audio, audio packets were written";
        return {};
    } catch (const std::exception& e) {
        return std::string("it does not open: ") + e.what();
    }
}

void PublishResult(const fs::path& partial, const fs::path& output, const ResultExpectation& expect) {
    if (const std::string why = CheckEncodedResult(partial, expect); !why.empty())
        Throw("process: the new result " + partial.string() + " is not right (" + why + "); it is left there, " + output.string() + " is as it was");
    std::error_code ec;
    Retrying(ec, 40, [&](std::error_code& e) { fs::rename(partial, output, e); });
    if (!ec) return;
    // Windows replaces no file another program has open without letting it be deleted - FFmpeg's readers, so most players (measured:
    // MoveFileExW and ReplaceFileW fail, a POSIX rename too). Overwriting it in place would not be atomic: a cancel or a fault in the
    // middle would lose both results. Both stay.
    const std::string why = ReadOnlyFile(output) ? "it is read-only: clear the attribute and rename the new file over it"
                                                 : "is it open in a player? Close it and rename the new file over it, or run again";
    Throw("process: the new result is " + partial.string() + " - " + output.string() + " could not be replaced (" + ec.message() + "): " + why);
}

bool ReadOnlyFile(const fs::path& file) {
    std::error_code ec;
    const fs::file_status st = fs::status(file, ec);  // MSVC: FILE_ATTRIBUTE_READONLY is a file without the write permissions
    return !ec && fs::is_regular_file(st) && (st.permissions() & fs::perms::owner_write) == fs::perms::none;
}

void RemovePartialResult(const fs::path& partial) {
    std::error_code ec;
    if (!fs::exists(partial, ec)) return;
    Retrying(ec, 10, [&](std::error_code& e) { fs::remove(partial, e); });
    if (ec) Log()->warn("process: cannot remove the unfinished result {}: {}", partial.string(), ec.message());
}

}  // namespace dlssvid
