#pragma once

#include <cstdint>
#include <filesystem>
#include <string>

namespace dlssvid {

// TASK-0070: the result of `process` is written next to its place and replaces the previous result in one rename, only when it is
// complete and reads back as what was encoded. A run cancelled (the GUI kills the CLI) or failed at any point leaves the previous result
// as it was.

// "<stem>.dlssvid-part<ext>" next to `output`: the extension stays, FFmpeg picks the container by it.
std::filesystem::path PartialResultPath(const std::filesystem::path& output);

struct ResultExpectation {
    uint32_t width = 0, height = 0;
    int64_t frames = -1;  // checked in MP4 and MOV, which count one sample per frame; < 0: not checked
    bool audio = false;   // audio packets were written (a track that got none is left out of MP4 and MOV)
};

// Empty when `file` opens as a video of that size, frame count and audio; else what differs.
std::string CheckEncodedResult(const std::filesystem::path& file, const ResultExpectation& expect);

// Renames the finished `partial` over `output`. Throws, `output` untouched and `partial` left where it is (the message says where), if
// `partial` does not check out or `output` cannot be replaced: Windows renames nothing over a file another program has open (a player;
// the GUI lets its viewer's file go while a run encodes), and overwriting it in place would not be atomic.
void PublishResult(const std::filesystem::path& partial, const std::filesystem::path& output, const ResultExpectation& expect);

// An existing file with the read-only attribute: nothing renames over it. `process` refuses such a result before it runs anything.
bool ReadOnlyFile(const std::filesystem::path& file);

// Removes `partial` of a run that ended without publishing it; quiet when there is none or it cannot go.
void RemovePartialResult(const std::filesystem::path& partial);

}  // namespace dlssvid
