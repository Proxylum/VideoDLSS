#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

#include "gpu/GpuFrame.h"
#include "io/SourceColor.h"
#include "pipeline/Frame.h"

struct AVFormatContext;
struct AVCodec;
struct AVCodecContext;
struct AVFrame;
struct AVPacket;
struct AVCodecParameters;
struct AVBufferRef;
struct SwsContext;

namespace dlssvid {

struct DecodeSupport;

struct Rational {
    int num = 0;
    int den = 1;
    double ToDouble() const { return den ? static_cast<double>(num) / den : 0.0; }
};

enum class HwAccel { None, Cuda };

struct VideoStreamInfo {
    // The frame as shown (TASK-0072): the stored frame turned by `orientation`, what NextFrame yields.
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t codedWidth = 0;   // the stored frame
    uint32_t codedHeight = 0;
    DisplayOrientation orientation;  // how players show the stream: its display matrix, else the 'rotate' tag
    std::string orientationFrom;     // "display matrix" | "rotate tag"; "" when not turned
    // TASK-0073: the proportions of a pixel - the container's, else the bitstream's (av_guess_sample_aspect_ratio); `codedSar` in the
    // stored grid's axes (as ffprobe prints it), `sar` in the axes of the turned frame NextFrame yields (the inverse for 90 / 270). The
    // frames keep the grid: `process` stretches its result to square pixels, passthrough writes the SAR.
    PixelAspect codedSar;
    PixelAspect sar;
    std::string sarFrom;  // "bitstream" (the codec says it; the container agreeing or silent) | "container"; "" when square
    Rational frameRate;    // av_guess_frame_rate
    Rational timeBase;     // stream time base (pts units)
    int64_t frameCount = 0;  // from container; 0 if unknown
    int64_t durationUs = 0;  // container duration in microseconds; 0 if unknown
    std::string codecName;
    std::string pixelFormat;  // source pixel format name (FFmpeg: "yuv420p", "yuv444p10le", ...)
    int bitDepth = 8;         // bits per sample of the source pixel format
    std::string chroma = "420";  // chroma subsampling of the source: "420" | "422" | "444" | "other"
    FrameDesc decoded;        // the CpuFrame layout NextFrame() yields for this stream (width/height set): the native
                              // planar YUV layout at the source's bit depth, or 8-bit 4:2:0 for formats that go through swscale;
                              // turned by `orientation` (OrientedDesc: 4:4:4 where the chroma grid does not map onto itself)
    int colorSpace = 2;       // AVColorSpace (2 = unspecified)
    int colorRange = 0;       // AVColorRange (0 = unspecified, 1 = limited/MPEG, 2 = full/JPEG)
    int colorPrimaries = 2;   // AVColorPrimaries (2 = unspecified)
    int colorTrc = 2;         // AVColorTransferCharacteristic (2 = unspecified)
    // TASK-0078 (HDR TB-1): the four fields above are the codec parameters after avformat_find_stream_info - the legacy path reads and
    // writes by them, unchanged. `color` is what the source's colour is: every layer (container, stream, first frame), the HDR records,
    // Dolby Vision, the class and its notes (io/SourceColor.h). Acted on from TB-2 (ResolveColor); `info` and the source card say it.
    SourceColor color;
    bool hasAudio = false;
    std::string audioCodecName;
    // TASK-0076: the decoder has no software path (this build's AV1, NvdecSupport.h): the frames come from NVDEC whatever
    // Options::hwaccel says, and CheckDecodeSupport says whether this GPU takes the stream
    bool hardwareOnly = false;
    bool nvdecCodec = false;  // NVDEC decodes this codec at all (HasNvdecHwaccel): HwAccel::Cuda can take it
};

// TASK-0076: `info` as a part of the stream decodes - a part in other parameters than the probe's (a new sequence header) or the layout
// the probe did not learn: the part's pixel format (its depth and chroma from FFmpeg's descriptor) and coded size. A format FFmpeg does not
// know (or "") and a size of 0 keep the probe's. CheckDecodeSupport of it judges the part as the probe was judged.
VideoStreamInfo StreamPartInfo(const VideoStreamInfo& info, const std::string& pixelFormat, uint32_t codedWidth, uint32_t codedHeight);

// TASK-0076: the part a refusal is about, as its why ends: «, on a part of the stream in other parameters (<fmt> <w>x<h>)» (a new
// sequence header) or «, in <fmt> <w>x<h> - a layout the probe did not learn» (the size the decoder judged it by - the probe's is the
// container's; read without it as well). VideoDecoder writes it; the GUI reads it back from another process's line (the proxy build)
// to judge the part as the decoder did.
struct StreamPart {
    std::string pixelFormat;  // "?" when the decoder did not say
    uint32_t codedWidth = 0, codedHeight = 0;  // 0: not said (a learned layout of the old words - the probe's size then)
    bool learned = false;     // the layout the probe did not learn, not a part in other parameters
};
std::string StreamPartWords(const StreamPart& part);
std::optional<StreamPart> StreamPartOf(const std::string& why);  // the last such words in `why`; nullopt: none

// TASK-0076: the payload of the (last) sequence header OBU in `data` - AV1's low-overhead format, as MP4 / MKV / WebM / IVF carry it, or
// an av1C record (its configOBUs after the 4-byte header). Empty: none, or the data does not read as OBUs
std::string_view Av1SequenceHeader(const uint8_t* data, size_t size);
// What FFmpeg's NVDEC makes its decoder for, from a sequence header's payload (AV1 5.5): av1dec sets the size and the software format
// from these (set_context_with_sequence, get_sw_pixel_format). Two headers with the same key decode on the same NVDEC decoder - a repeat
// that differs in operating_parameters_info only (AV1 7.5), a video joined to one like it. nullopt: the header does not parse
struct Av1SequenceKey {
    uint32_t profile = 0, bitDepth = 8, subsamplingX = 1, subsamplingY = 1, maxWidth = 0, maxHeight = 0;
    bool mono = false, rgb = false;  // rgb: BT.709 primaries, sRGB transfer, identity matrix (4:4:4 RGB)
    bool operator==(const Av1SequenceKey&) const = default;
};
std::optional<Av1SequenceKey> Av1SequenceKeyOf(std::string_view sequenceHeader);
// TASK-0076: what a chroma "other" is (VideoStreamInfo::chroma): "palette" (pal8), "rgb" (gbrp, bgr0 ...), "gray" (one component), a
// YUV as its J:a:b digits ("420", "444"; for a chroma "other": "411", "440", "410"), else "other" (an unknown format) - for the words
// a GUI says it in
std::string LayoutKindOf(const std::string& pixelFormat);

struct DecodeSupport;
// TASK-0076: a verdict a probe gives (the layout, no CUDA / NVDEC / AV1, the size, an odd height of a stream whose layout it learned) - not
// one a decoder met: on a part, in a layout the probe did not learn (its words), NVDEC refusing frames, frames of another size. A file
// written over is judged by its own probe against the first kind only (AppModel::refreshPlan)
bool IsProbeVerdict(const DecodeSupport& support);

// TASK-0076: a decoded frame smaller than the maximum its sequence header declares (an AV1 frame size override, a resize within the
// sequence) - FFmpeg 7.1.1's NVDEC makes its surfaces for the maximum and hands such a frame out at that size, a lower one also with the
// chroma mapped by its own height (refused; a narrower one of the full height out of caution - not seen on a GPU: no clip has an
// override). false: the header does not parse
bool Av1FrameBelowSequence(int width, int height, std::string_view sequenceHeader);

struct DecodeSupport;
// TASK-0076: the verdict another process met on this stream (the proxy build's «the video cannot be decoded here: <why>»), read back
// with its own block: a part or a layout the probe did not learn (StreamPartOf - judged again here), frames smaller than the sequence
// (Resized), an odd frame height (OddHeight) - RefusalText's words for that size -, else NVDEC refusing the stream (NvdecFailed). The why
// says `where` it was met
DecodeSupport VerdictFromWhy(const VideoStreamInfo& probe, const std::string& why, const std::string& where);

// Keyframes of a video stream from its packets (demux only, no decoding; TASK-0040): the longest run of frames from
// a keyframe to the next one or to the end - what a seek into the stream may have to decode.
struct KeyframeScan {
    int64_t frames = 0;           // video packets
    int64_t keyframes = 0;
    int64_t maxInterval = 0;      // frames from a keyframe up to the next keyframe (or the end), the longest
    bool intraOnly() const { return frames > 0 && keyframes == frames; }
};
KeyframeScan ScanKeyframes(const std::filesystem::path& path);

// The index of the frame at `pts` (in `timeBase` units) at `rate`, counted from the stream's `startTime`: what a seek and the frame
// indices after it assume - a constant frame rate.
int64_t PtsToIndex(int64_t pts, int64_t startTime, Rational timeBase, Rational rate);

// Whether the indices a seek derives from pts are the frames' own up to frame `upTo` (PR-10: dlssvid nr --start on the video path).
// Demux only, no decoding: the pts of the video packets up to `upTo` and a margin for reordering, sorted into presentation order;
// frame k must carry the time of frame k. A variable frame rate (or a gap in the timestamps) does not, and SeekToFrame would land on
// another frame. `indexRate` as VideoDecoder::Options::indexRate ({0, 1} = the stream's guessed rate).
struct FrameIndexScan {
    int64_t frames = 0;     // video packets read
    int64_t firstBad = -1;  // the first frame whose pts names another index (or that has no pts nor dts); -1: none
    int64_t badIndex = -1;  // the index that frame's pts names (-1 when it has no pts nor dts)
    bool noRate = false;    // the stream has no frame rate: no index can be taken from pts (firstBad 0)
    bool Consistent() const { return firstBad < 0; }
    std::string Why() const;  // «frame 61 carries the time of frame 66», «frame 12 has no timestamp», «the stream has no frame rate»
};
FrameIndexScan ScanFrameIndex(const std::filesystem::path& path, int64_t upTo, Rational indexRate = {0, 1});
// SeekToFrame decodes forward instead of seeking when the target is at most this many frames ahead: the indices are counted then, not
// taken from pts.
inline constexpr int64_t kSequentialSeekFrames = 48;

// The CpuFrame layout a stream's pixel format decodes into (VideoStreamInfo::decoded for a format name): planar YUV
// 4:2:0 / 4:2:2 / 4:4:4 at 8, 10, 12 or 16 bits stays as it is, everything else (RGB, 4:1:1, gray ...) becomes 8-bit
// 4:2:0 through swscale. `width` / `height` are left at 0.
FrameDesc DecodedFrameDescFor(const std::string& pixelFormat);

// The source's own layout, as shown: DecodedFrameDescFor(pixelFormat) at the shown size - what the encoder's «auto» follows (a 4:2:2
// stream turned by 90 decodes as 4:4:4, its result stays 4:2:2; TASK-0072).
FrameDesc SourceLayout(const VideoStreamInfo& info);

// The orientation of a display matrix (AV_PKT_DATA_DISPLAYMATRIX: 3x3 int32, 16.16 / 2.30 fixed point) as FFmpeg's command line
// applies it (fftools: transpose clock / cclock_flip, hflip / vflip, cclock / clock_flip). Not a multiple of 90 deg or degenerate: the
// identity, `why` says so. The matrix's scale is not looked at (non-square pixels are another matter).
DisplayOrientation OrientationFromDisplayMatrix(const int32_t matrix[9], std::string* why = nullptr);
// The 'rotate' tag of older muxers: degrees clockwise ("90", "-90", "450"); not a whole multiple of 90: the identity, `why` says so.
DisplayOrientation OrientationFromRotateTag(std::string_view tag, std::string* why = nullptr);

// FFmpeg demuxer + video decoder producing tightly packed planar YUV CpuFrames in the source's own layout: 8-bit
// 4:2:0 sources as before (bit-exact with FFmpeg), 10 / 12 / 16-bit and 4:2:2 / 4:4:4 sources in the matching 16-bit
// or 8-bit planar format of Frame.h (MSB-aligned decoder outputs such as P010 are shifted to LSB alignment, so NVDEC
// and software decoding agree bit for bit). With HwAccel::Cuda the stream is decoded by NVDEC; an NV12 frame stays in
// CUDA memory (primary context) and is exposed as a GpuFrame, and the CPU copy is made only when asked for (bit-exact
// NV12 -> YUV420P). Audio packets are handed to the sink untouched for stream copy. A stream shown turned (a display matrix, the
// 'rotate' tag: a phone video) is turned on the CPU after the copy (OrientFrame) and has no GpuFrame (TASK-0072); so has a frame of an
// odd width (any codec on NVDEC: the on-device consumers take even rows; TASK-0076 - an AV1 / VP9 frame comes at its surface's even size
// and is cut to its own, an H.264 / HEVC one comes cut by FFmpeg).
// A codec with NVDEC and a software path (h264, hevc, vp9 ...) of an odd height decodes in software also with HwAccel::Cuda (TASK-0076:
// FFmpeg's NVDEC gives its chroma wrong) - no device then, UsingHwAccel() false; except HEVC 4:4:4 (NvdecOddHeightSafe): it stays on
// NVDEC and holds the device, with no GpuFrame (not NV12). A codec without NVDEC (ProRes, FFV1) holds the device whatever its height,
// as before, and hands out no GpuFrame.
// A decoder without a software path (this build's AV1, TASK-0076, NvdecSupport.h) decodes on NVDEC whatever the options say: with
// HwAccel::Cuda the device is created at the open, as for any codec; otherwise on the first NextFrame (StartNvdec reopens codec_ on it) -
// a Probe() open never creates one. A stream this GPU cannot decode is refused then (or at RequireDecodable) with DecodeUnavailable,
// naming the reason; NVDEC refusing a frame is kept and said again on every later NextFrame. Its frames go to the host as with any NVDEC decode; a GpuFrame is handed
// out only when HwAccel::Cuda asked for it (UsingHwAccel).

// The container's global metadata (tags) of a media file, keys lower-cased (Matroska stores COMMENT): `comment`, `dlssvid_settings`
// (PR-11). Throws when the file cannot be opened.
std::map<std::string, std::string> ReadContainerTags(const std::filesystem::path& path);

class VideoDecoder {
public:
    struct Options {
        HwAccel hwaccel = HwAccel::None;
        // not news of the file: its open, NVDEC start, odd-height and sequence-header lines at debug level - a probe for Info(), a decoder
        // of the proxy pool, the viewport's original opened again (TASK-0076: an NVDEC-only video while a task runs)
        bool probe = false;
        // Slice threads only (an intra-only stream such as the scrubbing proxy): frame threads hold the first frame after
        // a seek back by a few decodes, which is the whole cost of a jump in an intra stream (TASK-0040).
        bool sliceThreadsOnly = false;
        // Frame indices from pts at this rate instead of the stream's (a proxy written at the rate of its original:
        // the container's guess of the rate may differ, the indices must not). {0, 1} = the stream's.
        Rational indexRate{0, 1};
        // TASK-0078: a look at the first frame (or packet) for the colour where the tags hint at HDR (NeedsFirstLook, cached per file:
        // FirstLookCached). Off for a reader that needs no colour - the result check, the thumbnails.
        bool colorProbe = true;
    };
    static Options Probe() {
        Options o;
        o.probe = true;
        return o;
    }

    explicit VideoDecoder(const std::filesystem::path& path, const Options& options = {});
    ~VideoDecoder();

    VideoDecoder(const VideoDecoder&) = delete;
    VideoDecoder& operator=(const VideoDecoder&) = delete;

    const VideoStreamInfo& Info() const { return info_; }
    bool HasAudio() const { return audioStream_ >= 0; }
    const AVCodecParameters* AudioCodecParameters() const;
    Rational AudioTimeBase() const;
    // HwAccel::Cuda was asked and the device is up: NextFrame may hand out GpuFrames (NV12 frames from NVDEC, not turned, of an even width; a codec without
    // NVDEC - FFV1, ProRes - decodes in software with none). Fixed at the open: a hardware-only decoder (TASK-0076) decoding on NVDEC
    // without being asked does not turn it on - its consumers stay on host frames.
    bool UsingHwAccel() const { return gpuFrames_; }
    // A frame came from NVDEC (asked for and the codec has it, or a hardware-only decoder past its first frame).
    bool DecodesOnNvdec() const { return nvdecFrames_; }
    // A CUDA device is held (HwAccel::Cuda at the open - not for a codec with NVDEC and a software path of an odd height
    // NvdecOddHeightSafe does not take (it decodes in software), nor when the device could not be created; a hardware-only decoder holds it
    // whatever its height, also one then refused (OddHeight) - or StartNvdec - even one NVDEC then refused): closing the decoder lets it go.
    bool HoldsCudaDevice() const { return hwDevice_ != nullptr; }
    // TASK-0076: refuses at once (DecodeUnavailable) a stream this machine cannot decode - a command calls it before its heavy preparation
    // (hashing, a model, an NGX feature) instead of learning it on the first frame. A no-op for a decoder with a software path.
    void RequireDecodable() const;

    using AudioPacketSink = std::function<void(AVPacket*)>;
    void SetAudioPacketSink(AudioPacketSink sink) { audioSink_ = std::move(sink); }

    // Next decoded frame in presentation order. Returns false at end of stream.
    // `gpu` (optional): filled with the on-device NV12 frame when UsingHwAccel() and the frame came from NVDEC as NV12; it is
    // valid until the next NextFrame() call; asked for without HwAccel::Cuda it stays invalid and the CpuFrame is filled (TASK-0076). `cpu = false` skips the device->host copy (the
    // CpuFrame then only carries index/pts/desc) — only allowed when gpu is requested; a frame decoded in software then comes on the host
    // all the same (TASK-0076). When NVDEC
    // yields a layout other than NV12 (10/12-bit, 4:4:4), the frame is of an odd width, or the stream is shown turned, `gpu` stays invalid and the CpuFrame is
    // always filled (as Info().decoded: the source's bit depth and chroma layout, turned - 4:4:4 where the turn does not map the chroma
    // grid onto itself, OrientedDesc), whatever `cpu` says.
    bool NextFrame(CpuFrame& out, GpuFrame* gpu = nullptr, bool cpu = true);

    // Random access: positions the decoder so that the next NextFrame() returns `index`
    // (assumes a constant frame rate for the pts <-> index mapping). Decoding forward from
    // the current position is used when the target is close ahead. Returns false when the
    // frame does not exist.
    bool SeekToFrame(int64_t index);
    // Whether SeekToFrame(index) decodes forward from here (indices counted, any frame rate) rather than seeking by pts.
    bool SeekIsSequential(int64_t index) const { return !drained_ && index >= nextIndex_ && index - nextIndex_ <= kSequentialSeekFrames; }
    int64_t NextIndex() const { return nextIndex_; }
    bool Drained() const { return drained_; }
    // Positions the demuxer at the keyframe at or before frame `index` and resets the decoder;
    // the caller walks forward with NextFrame() (frames carry their pts-derived index).
    // Returns false when the frame does not exist.
    bool SeekToKeyframeBefore(int64_t index);
    // false from a seek until the first frame after it: NextIndex() is a placeholder then (TASK-0040).
    bool PositionKnown() const { return !seeked_ || decodedSinceSeek_; }
    int64_t FramesDecoded() const { return framesDecoded_; }  // frames received from the decoder, all of them
    int64_t Seeks() const { return seeks_; }

private:
    Rational IndexRate() const { return indexRate_.num > 0 && indexRate_.den > 0 ? indexRate_ : info_.frameRate; }
    void ReadColour(bool look, const RawTags& container);  // TASK-0078: info_.color, once the stream is known and codec_ open
    AVCodecContext* OpenCodec(AVBufferRef* device) const;  // a codec context for the video stream: on `device`, else software threads
    void StartNvdec();  // a hardware-only decoder before its first packet: check the GPU, create the device, reopen codec_ on it
    // CheckAv, with ENOSYS of a hardware-only decoder as the refusal it is (kept: refused_); pts - of the packet just sent, if any
    void CheckDecode(int err, const char* what, std::optional<int64_t> pts = std::nullopt);
    // TASK-0076: where in the stream a refused frame is, in a refusal's words: «on the first frame», «at frame N» (0-based, as --frame),
    // «on the first frame after a seek» (no index known) - by its pts after a seek, else the next index; not framesDecoded_
    std::string PlaceWords(std::optional<int64_t> pts) const;
    void FollowSequenceHeader();  // TASK-0076: packet_ starts another coded video sequence - codec_ opened again on NVDEC for it
    bool ReceiveFrame(CpuFrame& out, GpuFrame* gpu, bool cpu);
    int64_t IndexFromPts(int64_t pts) const;
    bool DiscardFrames(int64_t upTo);  // decode and drop frames until nextIndex_ == upTo
    void ConvertFrame(AVFrame* frame, CpuFrame& out);

    AVFormatContext* fmt_ = nullptr;
    AVCodecContext* codec_ = nullptr;
    const AVCodec* decoder_ = nullptr;
    std::string url_;
    bool probe_ = false, sliceThreadsOnly_ = false;
    bool hwOnly_ = false;        // TASK-0076: HardwareOnlyDecoder(decoder_)
    bool nvdecStarted_ = false;  // StartNvdec ran (the GPU checked, codec_ on the device)
    bool gpuFrames_ = false;     // UsingHwAccel()
    bool nvdecFrames_ = false;   // DecodesOnNvdec()
    // StartNvdec: the driver said it takes the stream, or could not be asked for a CUDA error (memory) - an ENOSYS on the first frame
    // then is a start failure, not a refusal of the stream
    bool nvdecStartTransient_ = false;
    std::shared_ptr<const DecodeSupport> refused_;  // NVDEC refused this stream: every later NextFrame says so again
    // TASK-0076: the sequence header (its OBU's payload) codec_ decodes under - the stream's av1C, then the last one fed. One in other
    // parameters (another Av1SequenceKeyOf; a header that does not parse - other bytes) opens codec_ again: FFmpeg 7.1.1's av1dec keeps the
    // NVDEC decoder of the first through a new sequence header NVDEC takes too (its size and depth those of the first). One with the same
    // key (a repeat that differs in operating_parameters_info only, a video like it joined) goes on with codec_
    std::string nvdecSequence_;
    int64_t instanceFrames_ = 0;  // frames from this codec_ (a start failure is its first frame's)
    // TASK-0076: this decoder decoded frames under another AV1 sequence header (FollowSequenceHeader opened codec_ again after them): a
    // refusal now is a part's - also when the probe did not learn the layout (nothing to compare the part's with)
    bool decodedOtherSequence_ = false;
    AVBufferRef* hwDevice_ = nullptr;
    AVFrame* frame_ = nullptr;
    AVFrame* swFrame_ = nullptr;
    AVFrame* hwKeep_ = nullptr;  // last CUDA frame kept alive for GpuFrame
    AVPacket* packet_ = nullptr;
    SwsContext* sws_ = nullptr;
    int videoStream_ = -1;
    int audioStream_ = -1;
    int64_t nextIndex_ = 0;
    bool eofSent_ = false;
    bool drained_ = false;
    bool warnedConversion_ = false;
    bool warnedNonNv12_ = false;  // NVDEC layout other than NV12: the GPU fast path is off for this stream
    bool warnedTurnedGpu_ = false;  // a turned stream under NVDEC: no GpuFrame
    bool warnedOddWidthGpu_ = false;  // TASK-0076: an odd width under NVDEC: no GpuFrame
    bool checkedFrameMatrix_ = false;  // the first frame's own display matrix (an orientation SEI) compared with the stream's
    CpuFrame unturned_;  // the frame as stored, before OrientFrame
    bool seeked_ = false;  // indices come from pts after a seek
    bool decodedSinceSeek_ = false;
    int64_t framesDecoded_ = 0, seeks_ = 0;
    Rational indexRate_{0, 1};
    VideoStreamInfo info_;
    AudioPacketSink audioSink_;
};

}  // namespace dlssvid
