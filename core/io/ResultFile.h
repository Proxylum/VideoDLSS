#pragma once

#include <cstdint>
#include <filesystem>
#include <string>

namespace dlssvid {

// TASK-0070: the result of `process` is written next to its place and replaces the previous result only when it is complete and
// reads back as what was encoded. A run cancelled (the GUI kills the CLI) or failed at any point leaves the previous result as it was.

// "<stem>.dlssvid-part<ext>" next to `output`: the extension stays, FFmpeg picks the container by it.
std::filesystem::path PartialResultPath(const std::filesystem::path& output);

struct ResultExpectation {
    uint32_t width = 0, height = 0;
    int64_t frames = -1;  // checked when the container counts its frames (MP4, MOV; not MKV); < 0: not checked
    bool audio = false;   // an audio stream was written
};

// Empty when `file` opens as a video of that size, frame count and audio; else what differs.
std::string CheckEncodedResult(const std::filesystem::path& file, const ResultExpectation& expect);

struct PublishReport {
    bool copied = false;  // the previous result was open in another program: it was overwritten by a copy, not replaced by a rename
};
// What `process` adds to its warnings then.
inline constexpr const char* kResultCopiedWarning = "the previous result was open in another program: the new one was copied over it";

// Puts the finished `partial` in place of `output`. Throws, `output` untouched and `partial` left for a look, if `partial` does not
// check out. Windows renames nothing over a file another program has open (a player, the GUI's viewer), whatever that program shares:
// such a result is overwritten by a copy when the disk has room for it - readers that let others write (FFmpeg's do) allow that. When
// the copy cannot be made either, it throws: the new result stays as `partial`, the message says where.
PublishReport PublishResult(const std::filesystem::path& partial, const std::filesystem::path& output, const ResultExpectation& expect);

// Removes `partial` of a run that ended without publishing it; quiet when there is none or it cannot go.
void RemovePartialResult(const std::filesystem::path& partial);

}  // namespace dlssvid
