#include "io/ResultFile.h"

#include <chrono>
#include <system_error>
#include <thread>

#include "io/VideoDecoder.h"
#include "util/Error.h"
#include "util/Log.h"

namespace dlssvid {

namespace fs = std::filesystem;

namespace {
constexpr uintmax_t kCopyMargin = 64ull << 20;  // free space the copy leaves at least

// A rename the file system refused for a moment: an antivirus opens a file that has just been written. A file held for good still fails.
template <class Op>
void Retrying(std::error_code& ec, Op op) {
    for (int attempt = 0;; ++attempt) {
        ec.clear();
        op(ec);
        if (!ec || attempt >= 10) return;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
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
        if (expect.frames >= 0 && info.frameCount > 0 && info.frameCount != expect.frames)
            return "it holds " + std::to_string(info.frameCount) + " frames, " + std::to_string(expect.frames) + " were encoded";
        if (expect.audio && !probe.HasAudio()) return "it has no audio, the audio was copied";
        return {};
    } catch (const std::exception& e) {
        return std::string("it does not open: ") + e.what();
    }
}

PublishReport PublishResult(const fs::path& partial, const fs::path& output, const ResultExpectation& expect) {
    if (const std::string why = CheckEncodedResult(partial, expect); !why.empty())
        Throw("process: the new result " + partial.string() + " is not right (" + why + "); it is left there, " + output.string() + " is as it was");
    std::error_code ec;
    Retrying(ec, [&](std::error_code& e) { fs::rename(partial, output, e); });
    if (!ec) return {};

    // Windows renames nothing over a file another program has open - the GUI's viewer, a player. One that lets others write (FFmpeg's
    // readers do) still lets a copy overwrite it. The copy empties the old file first: the disk needs room for the difference.
    const std::string renameError = ec.message();
    const fs::path dir = fs::absolute(output).parent_path();
    const auto kept = [&](const std::string& why) {
        Throw("process: the new result is " + partial.string() + " - " + output.string() + " could not be replaced (" + why +
              "); close the program that has it open and rename the file, or run again");
    };
    std::error_code sizeEc, spaceEc;
    const uintmax_t newSize = fs::file_size(partial, sizeEc);
    uintmax_t oldSize = fs::file_size(output, ec);
    if (ec) oldSize = 0;
    const fs::space_info space = fs::space(dir, spaceEc);
    if (sizeEc || spaceEc) kept(renameError);
    if (space.available + oldSize < newSize + kCopyMargin) kept(renameError + "; no room on the disk to copy it over");
    fs::copy_file(partial, output, fs::copy_options::overwrite_existing, ec);
    if (ec) kept(renameError + "; copying over it: " + ec.message());
    if (const std::string why = CheckEncodedResult(output, expect); !why.empty()) kept("the copy over it is not right: " + why);
    RemovePartialResult(partial);
    Log()->warn("process: {} is open in another program - the new result was copied over it", output.string());
    return PublishReport{true};
}

void RemovePartialResult(const fs::path& partial) {
    std::error_code ec;
    if (!fs::exists(partial, ec)) return;
    Retrying(ec, [&](std::error_code& e) { fs::remove(partial, e); });
    if (ec) Log()->warn("process: cannot remove the unfinished result {}: {}", partial.string(), ec.message());
}

}  // namespace dlssvid
