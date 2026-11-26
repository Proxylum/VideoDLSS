#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

struct AVFrame;
struct AVFrameSideData;
struct AVPacketSideData;

namespace dlssvid {

// TASK-0075 / TASK-0078 (HDR step TB-1): what a source's colour is - read from every layer that says it, classified, with notes. Nothing
// here changes how a source is read or written yet: ResolveColor (convert/ColorPlan.h) still reads VideoStreamInfo's four codecpar fields
// (ColorInfoFromStream the matrix and the range, the result's tags the primaries and the transfer); it acts on the class from TB-2 on.

// Where a colour tag or an HDR record of a source was read.
enum class TagFrom : uint8_t {
    None,
    Container,  // the stream's parameters as the demuxer read them (MP4 / MOV colr, Matroska Colour) - before avformat_find_stream_info
    Stream,     // the codec parameters after it: the bitstream's (H.264 / HEVC VUI, an AV1 sequence header) where the decoder set them
    Frame,      // the first decoded frame (ProRes frame header, an HEVC alternative transfer SEI) - only where it differs from the stream
    Packet,     // the first video packet's side data (HDR10+ in Matroska)
};
std::string_view ToString(TagFrom from);  // "", "container", "stream", "frame", "first packet"

// The AVColor* values of one layer (2 / 2 / 2 / 0: unspecified).
struct RawTags {
    int matrix = 2;     // AVColorSpace
    int primaries = 2;  // AVColorPrimaries
    int trc = 2;        // AVColorTransferCharacteristic
    int range = 0;      // AVColorRange
    int chromaLoc = 0;  // AVChromaLocation
    bool operator==(const RawTags&) const = default;
};

struct Ratio {  // AVRational as read (not reduced)
    int num = 0;
    int den = 1;
    double Value() const { return den ? static_cast<double>(num) / den : 0.0; }
    bool operator==(const Ratio&) const = default;
};
// AVMasteringDisplayMetadata: the display a PQ / HLG master was graded on.
struct Mdcv {
    Ratio primaries[3][2];  // R, G, B: x, y
    Ratio white[2];
    Ratio minLum, maxLum;  // cd/m2
    bool hasPrimaries = false, hasLuminance = false;
    bool operator==(const Mdcv&) const = default;
};
struct Cll {  // AVContentLightMetadata (cd/m2)
    unsigned maxCll = 0, maxFall = 0;
    bool operator==(const Cll&) const = default;
};
struct Amve {  // AVAmbientViewingEnvironment
    Ratio illuminance, lightX, lightY;
    bool operator==(const Amve&) const = default;
};
struct HdrStatic {
    std::optional<Mdcv> mdcv;
    std::optional<Cll> cll;
    std::optional<Amve> amve;
    bool Empty() const { return !mdcv && !cll && !amve; }
    bool operator==(const HdrStatic&) const = default;
};
// AVDOVIDecoderConfigurationRecord (dvcC / dvvC / Matroska's mapping): blCompat is dv_bl_signal_compatibility_id, 4 bits as written.
struct DoviConf {
    int major = 0, minor = 0, profile = 0, level = 0, blCompat = 0;
    bool rpu = false, el = false, bl = false;
};
// The header of the first frame's RPU (AVDOVIRpuDataHeader) - a stream without a record (raw .hevc).
struct DoviRpu {
    int vdrRpuProfile = -1, vdrBitDepth = 0;
    bool blFullRange = false, elResampling = false, disableResidual = false;
};

enum class SourceClass : uint8_t {
    Legacy,  // SDR BT.709 / BT.601 (and anything else the table does not name): read as before
    Pq,      // trc 16 (SMPTE ST 2084)
    Hlg,     // trc 18 (ARIB STD-B67)
    Wide,    // BT.2020 primaries (or matrix) with an SDR curve
    Log,     // BT.2020 primaries with a curve that is not converted (camera log)
};
std::string_view ToString(SourceClass c);  // "sdr", "pq", "hlg", "wide", "log"

// What the source cannot get (acted on from TB-2; TB-1 only says it).
enum class ColourRefusal : uint8_t {
    None,
    Matrix,          // matrix 10 (BT.2020 CL) or 14 (ICtCp): no processing
    DolbyVisionIpt,  // a Dolby Vision base layer in IPT (profiles 5, 10.0; matrix 15): wrong colours without the RPU anywhere
    RgbHdr,          // an RGB HDR / BT.2020 stream: swscale makes it 8-bit 4:2:0 - wrong anywhere
};
std::string_view ToString(ColourRefusal r);  // "", "matrix", "dolby vision ipt", "rgb hdr"

// A note about the source. `key` names it for every language ("source.dv_removed"); `detail` is the English words of `dlssvid info`.
// plan = false: a difference of no consequence (the same class from two layers) - only `info` says it; plan = true: the plan and the
// result say it too (from TB-2).
struct ColourNote {
    std::string key;
    std::string detail;
    bool plan = true;
};

enum class FirstLook : uint8_t { None, Decode, Packet };
std::string_view ToString(FirstLook how);  // "none", "first frame", "first packet"

struct SourceColor {
    RawTags container;  // TagFrom::Container (all 2: the container says nothing)
    RawTags stream;     // TagFrom::Stream - VideoStreamInfo::colorSpace ... colorTrc, by which the legacy path reads and writes
    RawTags frame;      // the first frame's (all 2 when not decoded)
    RawTags effective;  // MergeTags of the three
    TagFrom matrixFrom = TagFrom::None, primariesFrom = TagFrom::None, trcFrom = TagFrom::None, rangeFrom = TagFrom::None;
    HdrStatic hdr;
    TagFrom mdcvFrom = TagFrom::None, cllFrom = TagFrom::None, amveFrom = TagFrom::None;
    std::optional<DoviConf> doviConf;
    std::optional<DoviRpu> doviRpu;
    int doviProfile = 0;  // the record's, else GuessDoviProfile of the RPU; 0: unknown or none
    bool hdr10Plus = false;
    TagFrom hdr10PlusFrom = TagFrom::None;  // Frame | Packet
    SourceClass cls = SourceClass::Legacy;     // by `effective`
    SourceClass rawCls = SourceClass::Legacy;  // ClassOf(stream): the fields the legacy path reads
    bool legacyPath = true;                    // cls and rawCls Legacy, nothing refused: read and written as before
    bool rgbLayout = false;                    // the stream's pixel format is RGB (decoded through swscale)
    bool iptAssumed = false;                   // DolbyVisionIpt for a base nobody names, no matrix nor transfer said: taken for IPT, not known
    ColourRefusal refuseProcessing = ColourRefusal::None;  // processing (stages) refused; passthrough allowed
    ColourRefusal refuseAll = ColourRefusal::None;         // processing and passthrough refused
    FirstLook looked = FirstLook::None;                    // what the first look read (None: not asked or failed)
    FirstLook lookFailed = FirstLook::None;                // the look that failed (Decode: the first frame; Packet: the first packet)
    std::string probeError;                                // the first look failed: why
    std::vector<ColourNote> notes;

    const ColourNote* Note(std::string_view key) const;  // nullptr: none
    bool Hdr() const { return cls == SourceClass::Pq || cls == SourceClass::Hlg; }
};

// The tags of three layers into one: per field the first specified one, frame -> stream -> container. A frame's value is attributed to
// the frame only where it differs from the stream's (FFmpeg fills a frame's unspecified fields from the codec context - the stream:
// libavcodec/decode.c fill_frame_props); the range is taken from the stream and the container only (ProRes frames always say limited -
// proresdec.c - and H.264 / HEVC / FFV1 frames repeat the stream's). `from` (optional): matrix, primaries, trc, range.
RawTags MergeTags(const RawTags& frame, const RawTags& stream, const RawTags& container, TagFrom from[4] = nullptr);

// The class of a tag set by the truth table's rows 4-10 (the design of TASK-0075, its truth table): trc 16 PQ, trc 18 HLG,
// primaries 9 Wide (a curve 1 / 6 / 13 / 14 / 15 / unspecified) or Log (any other curve), matrix 9 under other primaries Wide; else
// Legacy.
SourceClass ClassOf(int matrix, int primaries, int trc);
inline SourceClass ClassOf(const RawTags& t) { return ClassOf(t.matrix, t.primaries, t.trc); }

// Fills cls, rawCls, legacyPath, the refusals, doviProfile and the notes from the fields above (effective and stream set, the HDR records
// read). `layoutKind`: LayoutKindOf(pixel format) ("rgb" for an RGB stream); `codec`: the decoder's name ("av1": its HDR records live in
// the bitstream and are not read).
void Classify(SourceColor& c, std::string_view layoutKind, std::string_view codec);

// FFmpeg 7.1.1's guess of a Dolby Vision profile from an RPU header (libavcodec/dovi_rpu.c ff_dovi_guess_profile_hevc): 5, 7, 4, 8; 0
// unknown.
int GuessDoviProfile(const DoviRpu& rpu);

// Whether a source's colour needs a look at its first frame or packet. Decode: h264 / hevc / prores (the colour of a ProRes frame lives
// in its header, an HEVC SEI changes the transfer) with any hint of HDR or wide gamut in `tags` (the stream's over the container's), a
// raw elementary stream (demuxer "hevc" / "h264": no container), a 10-bit HEVC without tags, or a ProRes whose stream and container do
// not say all three (colr is optional; the frame header always says them). Packet: PQ in Matroska / WebM (HDR10+ rides in a packet's
// BlockAdditional). None: everything else - SDR 709 / 601 with its tags pays nothing.
FirstLook NeedsFirstLook(std::string_view codec, std::string_view demuxer, int bitDepth, const RawTags& tags);

// What a first look read.
struct FirstLookColor {
    FirstLook how = FirstLook::None;
    RawTags tags;  // the frame's (Decode)
    HdrStatic hdr;  // the frame's records (Decode)
    std::optional<DoviRpu> rpu;
    bool hdr10Plus = false;
    std::string error;  // the look failed (no frame, a decoder error): why
};

// Reserved values of the tags as unspecified (2): primaries and transfer 0 and 3 (H.273's reserved), matrix 3; for ProRes (`codec`
// "prores") matrix 0 as well - a ProRes header's 0 is «unknown» (FFmpeg's own `prores` encoder writes it for any value outside its lists),
// and ProRes is always Y'CbCr.
RawTags WithoutReserved(const RawTags& t, std::string_view codec);

// Two records of the same display or light, written as different fractions (Matroska's floats through av_d2q, an SEI's 1/50000 and
// 1/10000): equal at MP4's units - chromaticities to 1/50000, luminance and illuminance to 1/10000 - and with the same flags.
bool SameRecord(const Mdcv& a, const Mdcv& b);
bool SameRecord(const Amve& a, const Amve& b);

// What was read of a source, before it is merged: the layers' tags, the records of the container (codecpar's coded side data) and of the
// codec (its decoded_side_data: an HEVC hvcC's SEI), the Dolby Vision record, the first look when one was asked.
struct ColourLayers {
    RawTags container, stream;
    HdrStatic containerRecords, codecRecords;
    std::optional<DoviConf> dovi;
    std::optional<FirstLookColor> first;
};
// The source's colour from them: reserved values dropped (WithoutReserved), the tags merged (MergeTags), the records - the container's,
// the codec's where the container has none, the first frame's where it has one of another value (SameRecord) -, the RPU and HDR10+ of
// the look, a failed look as probeError / lookFailed; then Classify.
SourceColor SourceColorOf(const ColourLayers& in, std::string_view layoutKind, std::string_view codec);

// A field two layers say differently: "transfer" (FFmpeg's names of the values), `a` the first layer's value, `b` the second's.
struct TagDifference {
    std::string_view field;  // "matrix", "primaries", "transfer"
    int a = 2, b = 2;
};
// The first frame against the stream: fields both say, differently - not an HLG transfer (18) over the SDR curve an HEVC VUI carries with
// an alternative transfer SEI (1 / 6 / 14 / 15).
std::vector<TagDifference> FrameDifferences(const SourceColor& c);
// The container against what was taken from the stream or the frame: fields the container says, another layer decided otherwise, and
// the stream does not say as the container does (that difference is the frame's: FrameDifferences / layers_differ); the same HLG
// exception.
std::vector<TagDifference> ContainerDifferences(const SourceColor& c);
// Where the effective tags that differ from the codec parameters were read: "the first frame says", "the container says", "the first
// frame and the container say" (the words of source.layers_differ)
std::string_view LayersWords(const SourceColor& c);
// The value a TagDifference names in FFmpeg's words ("arib-std-b67")
std::string TagValueName(std::string_view field, int value);
// The first look at a file, once per (absolute path, size, modification time) in this process: a software decode of the first frame
// (slice threads, at most 64 video packets, then a flush) or a read of the first video packet. Packet is answered by a cached Decode too.
// A file modified less than 2 s before its look is looked at again (as Sha256FileCached; a modification time ahead of the clock by more
// than that counts as settled). Failures are answered (error) and cached, not thrown. At most 64 files, the least recently used dropped.
FirstLookColor FirstLookCached(const std::filesystem::path& path, FirstLook how);
int64_t FirstFrameDecodes();  // looks that decoded a frame (tests: the cache)
int64_t FirstPacketPeeks();   // looks that read a packet

RawTags FrameTagsOf(const AVFrame* frame);
// The HDR records among a stream's coded side data (AVCodecParameters / AVPacket): mastering display, light level, ambient viewing
// environment, the Dolby Vision record; fields absent there are left as they are.
void ReadCodedSideData(const AVPacketSideData* sd, int count, HdrStatic& hdr, std::optional<DoviConf>* dovi = nullptr);
// The same among frame side data (an AVFrame's, an AVCodecContext's decoded_side_data); `rpu`: the Dolby Vision RPU header, `hdr10Plus`:
// HDR10+ dynamic metadata present.
void ReadFrameSideData(AVFrameSideData* const* sd, int count, HdrStatic& hdr, std::optional<DoviRpu>* rpu = nullptr, bool* hdr10Plus = nullptr);

// The lines `dlssvid info` prints about the colour, each "<label>: <text>" with the label padded as info's ("colour:      HDR HLG - ...",
// "mastering:", "light level:", "ambient:", "dolby vision:", "hdr10+:", "note:"). None for a source on the legacy path with nothing to say
// (no HDR record, no Dolby Vision, no note for the plan): an SDR clip prints as before.
std::vector<std::string> ColourInfoLines(const SourceColor& c);
std::string_view ClassWords(SourceClass c);  // "SDR", "HDR PQ", "HDR HLG", "wide gamut SDR", "log curve"

// "matrix bt2020nc, primaries bt2020, transfer smpte2084, range tv" (FFmpeg's names: "unknown" for 2, and for range 0; a number where
// FFmpeg has no name).
std::string ColourWords(const RawTags& t);
std::string MatrixName(int matrix);
std::string PrimariesName(int primaries);
std::string TrcName(int trc);
std::string RangeName(int range);

}  // namespace dlssvid
