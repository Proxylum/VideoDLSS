#include "io/ResultFile.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <chrono>
#include <system_error>
#include <thread>

#include "io/VideoDecoder.h"
#include "util/Error.h"
#include "util/Log.h"
#include "util/Paths.h"

namespace dlssvid {

namespace fs = std::filesystem;

namespace {
// The partial file's name is 13 characters longer than the result's: FFmpeg writes paths past MAX_PATH by itself (its `\\?\` prefix),
// the std::filesystem calls here need it too, as absolute paths (a relative -o)
fs::path Long(const fs::path& p) { return Win32LongPath(fs::absolute(p)); }

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
    // NTFS takes at most 255 UTF-16 units in one name: a result name of 243 or more would make the partial one impossible - its
    // stem is cut instead (not inside a surrogate pair)
    constexpr size_t kMaxName = 255;
    std::wstring stem = output.stem().wstring();
    const std::wstring tail = L".dlssvid-part" + output.extension().wstring();
    if (stem.size() + tail.size() > kMaxName) {
        stem.resize(tail.size() < kMaxName ? kMaxName - tail.size() : 0);
        if (!stem.empty() && stem.back() >= 0xD800 && stem.back() <= 0xDBFF) stem.pop_back();
    }
    return output.parent_path() / (stem + tail);
}

std::string CheckEncodedResult(const fs::path& file, const ResultExpectation& expect) {
    try {
        VideoDecoder probe(file, VideoDecoder::Probe());
        const VideoStreamInfo& info = probe.Info();
        // the frames are written upright (TASK-0072): a display matrix on the result would turn them again in a player
        if (!info.orientation.Identity())
            return "it carries a display orientation (" + info.orientation.Token() + "): the frames were written upright, a player would turn them again";
        // TASK-0073: the pixels as written - square for a result stretched to them, the source's SAR in passthrough. A muxer that keeps
        // the frame as shown in whole pixels (WebM's DisplayWidth: 32:27 of 720 comes back 853:720) wrote the same picture
        const bool sameShown = !expect.sar.Square() && !info.codedSar.Square() &&
                               std::fabs(info.codedSar.Value() - expect.sar.Value()) * static_cast<double>(info.width) <= 1.0;
        if (!(info.codedSar == expect.sar) && !sameShown) {
            if (expect.sar.Square())
                return "it carries a pixel aspect of " + info.codedSar.Token() + " (square pixels were written): players would stretch it";
            return "its pixels are " + (info.codedSar.Square() ? std::string("square") : info.codedSar.Token()) + ", " + expect.sar.Token() +
                   " was written: players would show it in other proportions";
        }
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
    Retrying(ec, 40, [&](std::error_code& e) { fs::rename(Long(partial), Long(output), e); });
    if (!ec) return;
    // Windows replaces no file another program has open (measured: MoveFileExW, which fs::rename calls, fails even when the reader
    // shares delete; ReplaceFileW and a POSIX rename fail when it does not - FFmpeg's readers, so most players). Overwriting it in
    // place would not be atomic: a cancel or a fault in the middle would lose both results. Both stay.
    // dlssvid's own viewer shows the old result again when the run has ended: running again once the player is closed always works
    const std::string why = ReadOnlyFile(output) ? "it is read-only: clear the attribute and rename the new file over it"
                                                 : "is it open in a player? Run again once it is closed, or close every program that shows it "
                                                   "(dlssvid too: close the project) and rename the new file over it";
    Throw("process: the new result is " + partial.string() + " - " + output.string() + " could not be replaced (" + ec.message() + "): " + why);
}

bool ReadOnlyFile(const fs::path& file) {
    std::error_code ec;
    const fs::file_status st = fs::status(Long(file), ec);  // MSVC: FILE_ATTRIBUTE_READONLY is a file without the write permissions
    return !ec && fs::is_regular_file(st) && (st.permissions() & fs::perms::owner_write) == fs::perms::none;
}

PartialResult::PartialResult(fs::path partial) : path_(std::move(partial)) {
    std::error_code ec;
    existed_ = fs::exists(Long(path_), ec);
    if (existed_) time_ = fs::last_write_time(Long(path_), ec);
}

void PartialResult::DiscardIfWritten() const {
    std::error_code ec;
    if (existed_ && fs::exists(Long(path_), ec) && fs::last_write_time(Long(path_), ec) == time_ && !ec) return;  // not touched by this run
    RemovePartialResult(path_);
}

void RemovePartialResult(const fs::path& partial) {
    std::error_code ec;
    if (!fs::exists(Long(partial), ec)) return;
    Retrying(ec, 10, [&](std::error_code& e) { fs::remove(Long(partial), e); });
    if (ec) Log()->warn("process: cannot remove the unfinished result {}: {}", partial.string(), ec.message());
}

}  // namespace dlssvid
