// dlssvid — command line front-end. Everything the GUI can do must be reachable from here (ТЗ §7).
// Stage 0: `info`, `process --passthrough`. Stage 1: `export`, `import`, `convert`.

#include <CLI/CLI.hpp>

#include <cstdio>
#include <string>

#include "convert/ColorConvert.h"
#include "convert/DepthConvert.h"
#include "convert/MvConvert.h"
#include "gpu/D3D12Device.h"
#include "io/VideoDecoder.h"
#include "io/VideoEncoder.h"
#include "passes/PassSequence.h"
#include "pipeline/Pipeline.h"
#include "stages/passthrough/PassthroughStage.h"
#include "util/Error.h"
#include "util/Log.h"
#include "util/Sha256.h"

#ifdef DLSSVID_WITH_CUDA
#include "gpu/CudaInterop.h"
#endif

using namespace dlssvid;

namespace {

// ---- helpers ------------------------------------------------------------------------------

void PrintProgress(int64_t done, int64_t total) {
    if (done % 25 == 0 || done == total) {
        if (total > 0) std::fprintf(stderr, "\r%lld/%lld frames", static_cast<long long>(done), static_cast<long long>(total));
        else std::fprintf(stderr, "\r%lld frames", static_cast<long long>(done));
    }
}

FrameRange ParseRangeOrThrow(const std::string& s) {
    if (s.empty()) return FrameRange{};
    const auto r = FrameRange::Parse(s);
    if (!r) Throw("--range expects a-b, a- or a (got '" + s + "')");
    return *r;
}

bool ParseSize(const std::string& s, uint32_t& w, uint32_t& h) {
    if (s.empty()) return false;
    const size_t x = s.find('x');
    if (x == std::string::npos) Throw("--size / --target expects WxH (got '" + s + "')");
    w = static_cast<uint32_t>(std::stoul(s.substr(0, x)));
    h = static_cast<uint32_t>(std::stoul(s.substr(x + 1)));
    return true;
}

PassReader OpenReader(const std::string& dir, const std::string& passName, const std::string& convention, const std::string& size) {
    if (Manifest::Exists(dir) && passName.empty()) return PassReader::Open(dir);
    if (passName.empty()) Throw("no manifest.json in " + dir + ": specify --pass (and --convention, --size for raw dumps)");
    PassReader::ScanHints hints;
    const auto kind = ParsePassKind(passName);
    if (!kind) Throw("unknown pass: " + passName);
    hints.kind = *kind;
    if (!convention.empty()) {
        const auto c = ParseConvention(convention);
        if (!c) Throw("--convention must be raw|dlss");
        hints.convention = *c;
    } else {
        hints.convention = Spec(*kind).convention;
    }
    ParseSize(size, hints.width, hints.height);
    return PassReader::OpenWithoutManifest(dir, hints);
}

void PrintManifest(const PassReader& r) {
    const Manifest& m = r.Man();
    std::printf("dir:         %s%s\n", r.Dir().string().c_str(), r.HadManifest() ? "" : "  (no manifest, scanned)");
    std::printf("pass:        %s (%s)\n", m.pass.c_str(), std::string(ToString(m.convention)).c_str());
    std::printf("size:        %ux%u, %s x%zu, %s\n", m.width, m.height, std::string(ToString(m.pixelType)).c_str(), m.channels.size(),
                std::string(ToString(m.format)).c_str());
    std::printf("frames:      %lld [%lld..%lld]%s\n", static_cast<long long>(m.frameCount), static_cast<long long>(m.firstFrame),
                static_cast<long long>(m.lastFrame), r.MissingFrames().empty() ? "" : "  MISSING FRAMES");
    if (m.fps.num) std::printf("fps:         %d/%d\n", m.fps.num, m.fps.den);
    if (!m.colorspace.empty()) std::printf("colorspace:  %s\n", m.colorspace.c_str());
    if (m.pass == "depth_raw" || m.pass == "depth_dlss")
        std::printf("depth:       %s near=%g far=%g%s\n", m.depth.units.c_str(), m.depth.zNear, m.depth.zFar,
                    m.depth.relative ? (" min=" + std::to_string(m.depth.minValue) + " max=" + std::to_string(m.depth.maxValue)).c_str() : "");
    if (m.pass == "mv_raw" || m.pass == "mv_dlss") std::printf("mv:          %s, y_up=%d\n", m.mv.direction.c_str(), m.mv.yUp ? 1 : 0);
    if (!m.model.empty()) std::printf("model:       %s %s\n", m.model.c_str(), m.modelVersion.c_str());
    if (!m.sourceHash.empty()) std::printf("source:      %s %s\n", m.sourceFile.c_str(), m.sourceHash.c_str());
}

// ---- info -----------------------------------------------------------------------------------

int CmdInfo(const std::string& input, bool warp) {
    VideoDecoder dec(input);
    const auto& i = dec.Info();
    const ColorInfo ci = ColorInfoFromStream(i);
    std::printf("file:        %s\n", input.c_str());
    std::printf("video:       %s %ux%u %s (%s, %s range)\n", i.codecName.c_str(), i.width, i.height, i.pixelFormat.c_str(),
                std::string(ToString(ci.matrix)).c_str(), std::string(ToString(ci.range)).c_str());
    std::printf("frame rate:  %d/%d (%.3f fps)\n", i.frameRate.num, i.frameRate.den, i.frameRate.ToDouble());
    std::printf("time base:   %d/%d\n", i.timeBase.num, i.timeBase.den);
    std::printf("frames:      %lld%s\n", static_cast<long long>(i.frameCount), i.frameCount ? "" : " (unknown)");
    std::printf("duration:    %.3f s\n", i.durationUs / 1e6);
    std::printf("audio:       %s\n", i.hasAudio ? i.audioCodecName.c_str() : "none");

    D3D12Device device({warp, false});
    std::printf("d3d12:       %s%s, VRAM %llu MiB\n", device.AdapterName().c_str(), device.IsWarp() ? " [WARP]" : "",
                static_cast<unsigned long long>(device.DedicatedVideoMemory() / (1024 * 1024)));
#ifdef DLSSVID_WITH_CUDA
    std::string reason;
    if (!device.IsWarp() && CudaInterop::Available(&reason)) {
        try {
            CudaInterop cuda(device);
            std::printf("cuda:        device %d (%s)\n", cuda.CudaDeviceId(), cuda.CudaDeviceName().c_str());
        } catch (const std::exception& e) {
            std::printf("cuda:        unavailable (%s)\n", e.what());
        }
    } else {
        std::printf("cuda:        unavailable (%s)\n", device.IsWarp() ? "WARP adapter" : reason.c_str());
    }
#else
    std::printf("cuda:        not built\n");
#endif
    for (const char* name : {"h264_nvenc", "hevc_nvenc", "av1_nvenc", "ffv1"}) {
        std::printf("encoder:     %-10s %s\n", name, VideoEncoder::EncoderAvailable(name) ? "yes" : "no");
    }
    return 0;
}

// ---- process --------------------------------------------------------------------------------

struct ProcessArgs {
    std::string input;
    std::string output;
    bool passthrough = false;
    std::string codec = "h264_nvenc";
    std::vector<std::string> codecOptions;  // key=value
    std::string hwaccel = "none";
    int64_t frames = -1;
    bool warp = false;
    bool noGpuRoundTrip = false;
};

int CmdProcess(const ProcessArgs& a) {
    if (!a.passthrough) {
        Log()->error("only --passthrough is implemented so far (SR/NR/FG stages arrive in stages 5-7)");
        return 2;
    }
    VideoDecoder::Options dopt;
    if (a.hwaccel == "cuda") dopt.hwaccel = HwAccel::Cuda;
    else if (a.hwaccel != "none") Throw("--hwaccel must be none|cuda");

    VideoDecoder decoder(a.input, dopt);
    const auto& info = decoder.Info();

    VideoEncoder::Options eopt;
    eopt.codec = a.codec;
    eopt.frameRate = info.frameRate;
    eopt.timeBase = info.timeBase;
    for (const auto& kv : a.codecOptions) {
        const auto eq = kv.find('=');
        if (eq == std::string::npos) Throw("--codec-opt expects key=value, got: " + kv);
        eopt.codecOptions[kv.substr(0, eq)] = kv.substr(eq + 1);
    }
    VideoEncoder encoder(a.output, FrameDesc{info.width, info.height, PixelFormat::Yuv420p}, eopt);

    D3D12Device device({a.warp, false});
    Pipeline pipeline(device, 4);
    StageConfig cfg;
    cfg.name = "passthrough";
    cfg.params["gpu_roundtrip"] = !a.noGpuRoundTrip;
    pipeline.AddStage(std::make_unique<PassthroughStage>(), cfg);
    pipeline.Init();

    const int64_t total = a.frames > 0 ? a.frames : info.frameCount;
    const RunStats stats = RunPipeline(pipeline, decoder, encoder, [total](int64_t n) { PrintProgress(n, total); }, a.frames);
    std::fprintf(stderr, "\n");
    pipeline.Shutdown();

    Log()->info("done: {} frames in {:.2f} s ({:.1f} fps)", stats.framesOut, stats.seconds,
                stats.seconds > 0 ? stats.framesOut / stats.seconds : 0.0);
    return 0;
}

// ---- export ---------------------------------------------------------------------------------

struct ExportArgs {
    std::string input;    // video file (colour source)
    std::string fromDir;  // existing pass folder to re-export
    std::string output;
    std::string pass = "color_source";
    std::string format;   // exr|png16|tiff|npz|raw (default from preset / exr)
    std::string preset;   // nuke|comfyui|rawdlss
    std::string range;
    std::string depthDir, mvDir;  // for the nuke preset
    std::string matrix = "auto";  // bt601|bt709|auto
    std::string colorRange = "auto";
    std::string pixelType;  // override sample type
    std::string passName, convention, size;  // for --from-dir without manifest
};

int CmdExport(const ExportArgs& a) {
    if (a.input.empty() == a.fromDir.empty()) Throw("export: give exactly one of --input <video> or --from-dir <pass folder>");
    const auto preset = ParseExportPreset(a.preset);
    if (!preset) Throw("--preset must be nuke|comfyui|rawdlss");
    const FrameRange range = ParseRangeOrThrow(a.range);

    // Colour from a video file.
    std::unique_ptr<VideoDecoder> decoder;
    std::optional<PassReader> reader;
    Manifest base;
    FrameSource source;
    PassKind kind = PassKind::ColorSource;
    int64_t nextFrame = 0;

    if (!a.input.empty()) {
        decoder = std::make_unique<VideoDecoder>(a.input);
        const auto& info = decoder->Info();
        ColorInfo ci = ColorInfoFromStream(info);
        if (a.matrix == "bt601") ci.matrix = ColorMatrix::Bt601;
        else if (a.matrix == "bt709") ci.matrix = ColorMatrix::Bt709;
        else if (a.matrix != "auto") Throw("--matrix must be bt601|bt709|auto");
        if (a.colorRange == "limited") ci.range = ColorRange::Limited;
        else if (a.colorRange == "full") ci.range = ColorRange::Full;
        else if (a.colorRange != "auto") Throw("--color-range must be limited|full|auto");
        const auto k = ParsePassKind(a.pass);
        if (!k || !Spec(*k).isColor) Throw("from a video only colour passes can be exported (got '" + a.pass + "')");
        kind = *k;
        base = Manifest::ForPass(kind, info.width, info.height, FileFormat::Exr);
        base.fps = info.frameRate;
        base.sourceFile = std::filesystem::path(a.input).filename().string();
        base.sourceHash = "sha256:" + Sha256File(a.input);
        base.stageParams = {{"matrix", std::string(ToString(ci.matrix))}, {"range", std::string(ToString(ci.range))}};
        Log()->info("colour: {} {} range", ToString(ci.matrix), ToString(ci.range));
        // Sequential source: frames must be requested in increasing order (skips are decoded and dropped).
        source = [&, ci](int64_t frame, PassImage& out) {
            CpuFrame f;
            while (nextFrame <= frame) {
                if (!decoder->NextFrame(f)) return false;
                ++nextFrame;
            }
            out = Yuv420pToRgb(f, ci, PixelType::F16);
            return true;
        };
    } else {
        reader.emplace(OpenReader(a.fromDir, a.passName, a.convention, a.size));
        base = reader->Man();
        const auto k = base.Kind();
        if (!k) Throw("cannot re-export custom pass '" + base.pass + "'");
        kind = *k;
        source = SourceFromReader(*reader);
        if (a.range.empty()) {
            // default: whole folder
            const_cast<FrameRange&>(range).first = base.firstFrame;
            const_cast<FrameRange&>(range).last = base.lastFrame;
        }
    }

    // Preset "nuke": multi-layer EXR from colour + optional depth/mv folders.
    if (*preset == ExportPreset::Nuke) {
        if (!Spec(kind).isColor) Throw("the nuke preset needs a colour pass as the base (use --input or a colour --from-dir)");
        std::optional<PassReader> depth, mv;
        FrameSource depthSrc, mvSrc;
        if (!a.depthDir.empty()) {
            depth.emplace(PassReader::Open(a.depthDir));
            depth->Validate({base.width, base.height, -1, std::nullopt});
            depthSrc = SourceFromReader(*depth);
        }
        if (!a.mvDir.empty()) {
            mv.emplace(PassReader::Open(a.mvDir));
            mv->Validate({base.width, base.height, -1, std::nullopt});
            mvSrc = SourceFromReader(*mv);
        }
        const int64_t n = ExportLayeredExr(a.output, "layered", base, source, depthSrc ? &depthSrc : nullptr, mvSrc ? &mvSrc : nullptr,
                                           range, PrintProgress);
        std::fprintf(stderr, "\n");
        Log()->info("exported {} multi-layer EXR frames -> {}", n, a.output);
        return 0;
    }

    Manifest m = base;
    m.pass = Spec(kind).name;
    m.format = *preset != ExportPreset::None ? PresetFormat(*preset, kind) : FileFormat::Exr;
    m.pixelType = *preset != ExportPreset::None ? PresetPixelType(*preset, kind) : Spec(kind).type;
    if (!a.format.empty()) {
        const auto f = ParseFileFormat(a.format);
        if (!f) Throw("--format must be exr|png16|tiff|npz|raw");
        m.format = *f;
        if (*f == FileFormat::Png && m.pixelType != PixelType::U8 && m.pixelType != PixelType::U16)
            m.pixelType = kind == PassKind::Mask ? PixelType::U8 : PixelType::U16;
    }
    if (!a.pixelType.empty()) {
        const auto t = ParsePixelType(a.pixelType);
        if (!t) Throw("--pixel-type must be u8|u16|f16|f32");
        m.pixelType = *t;
    }
    if (*preset == ExportPreset::RawDlss && Spec(kind).convention != Convention::Dlss)
        Throw("the rawdlss preset expects depth_dlss / mv_dlss passes: run `dlssvid convert` first");
    m.channels = Spec(kind).channels;
    m.filePattern = m.pass + "_%06d" + ExtensionFor(m.format, m.pixelType, m.channels.size());
    if (Spec(kind).isColor && (m.pixelType == PixelType::U16 || m.pixelType == PixelType::U8) && base.Kind() && Spec(*base.Kind()).isColor &&
        base.pixelType != m.pixelType) {
        // colour float [0,1] -> integer scale
        const float scale = m.pixelType == PixelType::U16 ? 65535.f : 255.f;
        FrameSource inner = source;
        source = [inner, scale](int64_t f, PassImage& out) {
            if (!inner(f, out)) return false;
            if (out.type == PixelType::F16 || out.type == PixelType::F32) {
                for (uint32_t y = 0; y < out.height; ++y)
                    for (uint32_t x = 0; x < out.width; ++x)
                        for (size_t c = 0; c < out.channels.size(); ++c) out.Set(x, y, c, out.Get(x, y, c) * scale);
            }
            return true;
        };
    }
    const int64_t n = ExportPass(a.output, m, source, range, PrintProgress);
    std::fprintf(stderr, "\n");
    Log()->info("exported {} frames of {} as {} -> {}", n, m.pass, ToString(m.format), a.output);
    return 0;
}

// ---- import ---------------------------------------------------------------------------------

struct ImportArgs {
    std::string dir;
    std::string pass, convention, size;
    std::string expectSize;
    int64_t expectFrames = -1;
    std::string toDlss, toRaw;  // conversion output folders
    float zNear = 0.f, zFar = 0.f;
    bool relative = false;
    std::string depthDir;  // occlusion depth for mv conversion
    std::string target;    // WxH for mv_dlss
    int dilate = 1;
    bool writeManifest = false;
};

int ConvertFolder(const PassReader& in, const std::string& outDir, PassKind toKind, const ImportArgs& a);

int CmdImport(const ImportArgs& a) {
    PassReader r = OpenReader(a.dir, a.pass, a.convention, a.size);
    if (a.zNear > 0.f) r.Man().depth.zNear = a.zNear;
    if (a.zFar > 0.f) r.Man().depth.zFar = a.zFar;
    if (a.relative) {
        r.Man().depth.relative = true;
        r.Man().depth.units = "relative";
    }
    PrintManifest(r);
    PassReader::Expect expect;
    ParseSize(a.expectSize, expect.width, expect.height);
    expect.frameCount = a.expectFrames;
    r.Validate(expect);
    // Read one frame to prove the files decode.
    const PassImage first = r.ReadFrame(r.Man().firstFrame);
    std::printf("first frame: %ux%u %s x%zu ok\n", first.width, first.height, std::string(ToString(first.type)).c_str(), first.channels.size());
    if (a.writeManifest && !r.HadManifest()) {
        r.Man().Save(r.Dir());
        Log()->info("manifest.json written to {}", r.Dir().string());
    }
    const auto kind = r.Man().Kind();
    if (!a.toDlss.empty()) {
        if (kind == PassKind::DepthRaw) return ConvertFolder(r, a.toDlss, PassKind::DepthDlss, a);
        if (kind == PassKind::MvRaw) return ConvertFolder(r, a.toDlss, PassKind::MvDlss, a);
        Throw("--to-dlss needs a depth_raw or mv_raw pass");
    }
    if (!a.toRaw.empty()) {
        if (kind == PassKind::DepthDlss) return ConvertFolder(r, a.toRaw, PassKind::DepthRaw, a);
        if (kind == PassKind::MvDlss) return ConvertFolder(r, a.toRaw, PassKind::MvRaw, a);
        Throw("--to-raw needs a depth_dlss or mv_dlss pass");
    }
    return 0;
}

// ---- convert --------------------------------------------------------------------------------

int ConvertFolder(const PassReader& in, const std::string& outDir, PassKind toKind, const ImportArgs& a) {
    const Manifest& src = in.Man();
    const auto fromKind = src.Kind();
    if (!fromKind) Throw("cannot convert custom pass '" + src.pass + "'");
    Manifest out = Manifest::ForPass(toKind, src.width, src.height, src.format == FileFormat::Png ? FileFormat::Exr : src.format);
    if (out.format == FileFormat::Npz) out.format = FileFormat::Npz;
    out.fps = src.fps;
    out.sourceFile = src.sourceFile;
    out.sourceHash = src.sourceHash;
    out.model = src.model;
    out.modelVersion = src.modelVersion;
    out.stageParams = src.stageParams;
    out.depth = src.depth;
    out.mv = src.mv;
    out.mv.direction = toKind == PassKind::MvDlss ? "backward" : "forward";
    if (a.zNear > 0.f) out.depth.zNear = a.zNear;
    if (a.zFar > 0.f) out.depth.zFar = a.zFar;
    if (a.relative) {
        out.depth.relative = true;
        out.depth.units = "relative";
    }
    out.filePattern = out.pass + "_%06d" + ExtensionFor(out.format, out.pixelType, out.channels.size());

    std::optional<PassReader> depth;
    if (!a.depthDir.empty()) {
        depth.emplace(PassReader::Open(a.depthDir));
        depth->Validate({src.width, src.height, -1, std::nullopt});
    }
    MvConvertOptions mvOpt;
    mvOpt.dilateRadius = a.dilate;
    ParseSize(a.target, mvOpt.targetWidth, mvOpt.targetHeight);
    if (mvOpt.targetWidth) {
        out.width = mvOpt.targetWidth;
        out.height = mvOpt.targetHeight;
        out.mv.refWidth = mvOpt.targetWidth;
        out.mv.refHeight = mvOpt.targetHeight;
    }

    const int64_t total = src.lastFrame - src.firstFrame + 1;
    PassWriter writer(outDir, out);
    for (int64_t f = src.firstFrame; f <= src.lastFrame; ++f) {
        PassImage result;
        switch (toKind) {
            case PassKind::DepthDlss: {
                DepthParams p = writer.Man().depth;
                result = DepthRawToDlss(in.ReadFrame(f), p);
                if (f == src.firstFrame) {
                    // keep the range computed from the first frame for the whole sequence
                    const_cast<Manifest&>(writer.Man()).depth = p;
                }
                break;
            }
            case PassKind::DepthRaw: result = DepthDlssToRaw(in.ReadFrame(f), out.depth); break;
            case PassKind::MvDlss: {
                // mv_dlss[f] comes from mv_raw[f-1]; the first frame has no previous -> zeros
                if (f == src.firstFrame) {
                    result = ZeroMv(out.width, out.height);
                } else {
                    PassImage d;
                    if (depth && depth->HasFrame(f - 1)) d = depth->ReadFrame(f - 1);
                    result = ForwardFlowToBackwardMv(in.ReadFrame(f - 1), d.Empty() ? nullptr : &d, mvOpt);
                }
                break;
            }
            case PassKind::MvRaw: {
                // mv_raw[f] comes from mv_dlss[f+1]; the last frame has no next -> zeros
                if (f == src.lastFrame) {
                    result = ZeroMv(out.width, out.height);
                    result.channels = Spec(PassKind::MvRaw).channels;
                } else {
                    PassImage d;
                    if (depth && depth->HasFrame(f + 1)) d = depth->ReadFrame(f + 1);
                    result = BackwardMvToForwardFlow(in.ReadFrame(f + 1), d.Empty() ? nullptr : &d, mvOpt);
                }
                break;
            }
            default: Throw("convert: unsupported target pass");
        }
        writer.WriteFrame(f, result);
        PrintProgress(f - src.firstFrame + 1, total);
    }
    writer.Finish();
    std::fprintf(stderr, "\n");
    Log()->info("converted {} frames {} -> {} in {}", total, src.pass, out.pass, outDir);
    return 0;
}

struct ConvertArgs {
    ImportArgs in;  // reuse input options (dir, pass, near/far, depthDir, target, dilate)
    std::string output;
    std::string to;
};

int CmdConvert(const ConvertArgs& a) {
    const auto toKind = ParsePassKind(a.to);
    if (!toKind) Throw("--to must be depth_dlss|depth_raw|mv_dlss|mv_raw");
    PassReader r = OpenReader(a.in.dir, a.in.pass, a.in.convention, a.in.size);
    if (a.in.zNear > 0.f) r.Man().depth.zNear = a.in.zNear;
    if (a.in.zFar > 0.f) r.Man().depth.zFar = a.in.zFar;
    if (a.in.relative) {
        r.Man().depth.relative = true;
        r.Man().depth.units = "relative";
    }
    r.Validate({});
    return ConvertFolder(r, a.output, *toKind, a.in);
}

void AddImportOptions(CLI::App* cmd, ImportArgs& a) {
    cmd->add_option("-i,--input", a.dir, "pass folder")->required();
    cmd->add_option("--pass", a.pass, "pass name when there is no manifest (depth_raw, mv_raw, ...)");
    cmd->add_option("--convention", a.convention, "raw|dlss when there is no manifest");
    cmd->add_option("--size", a.size, "WxH for raw dumps without a manifest");
    cmd->add_option("--near", a.zNear, "near plane in metres (depth conversions)");
    cmd->add_option("--far", a.zFar, "far plane in metres (depth conversions)");
    cmd->add_flag("--relative", a.relative, "treat depth as relative (normalised onto [near, far])");
    cmd->add_option("--depth-dir", a.depthDir, "depth_raw folder used to resolve occlusions when converting motion vectors");
    cmd->add_option("--target", a.target, "WxH target resolution for mv_dlss");
    cmd->add_option("--dilate", a.dilate, "MV dilation radius at depth edges (0-2)")->default_val(1);
}

}  // namespace

int main(int argc, char** argv) {
    CLI::App app{"dlssvid — offline DLSS video pipeline"};
    app.set_version_flag("--version", DLSSVID_VERSION);
    app.require_subcommand(1);
    bool verbose = false;
    app.add_flag("-v,--verbose", verbose, "debug logging");

    std::string infoInput;
    bool infoWarp = false;
    auto* info = app.add_subcommand("info", "print stream and device information");
    info->add_option("input", infoInput, "input video file")->required()->check(CLI::ExistingFile);
    info->add_flag("--warp", infoWarp, "use the WARP software adapter");

    ProcessArgs pa;
    auto* process = app.add_subcommand("process", "run the pipeline on a video file");
    process->add_option("-i,--input", pa.input, "input video file")->required()->check(CLI::ExistingFile);
    process->add_option("-o,--output", pa.output, "output video file (container by extension)")->required();
    process->add_flag("--passthrough", pa.passthrough, "decode -> GPU -> encode without processing");
    process->add_option("--codec", pa.codec, "video encoder: h264_nvenc | hevc_nvenc | av1_nvenc | ffv1")->default_val("h264_nvenc");
    process->add_option("--codec-opt", pa.codecOptions, "encoder private option key=value (repeatable)");
    process->add_option("--hwaccel", pa.hwaccel, "decoder hardware acceleration: none | cuda")->default_val("none");
    process->add_option("--frames", pa.frames, "process at most N frames")->default_val(-1);
    process->add_flag("--warp", pa.warp, "use the WARP software adapter");
    process->add_flag("--no-gpu-roundtrip", pa.noGpuRoundTrip, "skip the GPU upload/readback in passthrough");

    ExportArgs ea;
    auto* exp = app.add_subcommand("export", "export a pass as a file sequence + manifest.json");
    exp->add_option("-i,--input", ea.input, "source video (colour passes)")->check(CLI::ExistingFile);
    exp->add_option("--from-dir", ea.fromDir, "existing pass folder to re-export in another format/preset");
    exp->add_option("-o,--output", ea.output, "output folder")->required();
    exp->add_option("--pass", ea.pass, "pass to export (video: color_source)")->default_val("color_source");
    exp->add_option("--format", ea.format, "exr | png16 | tiff | npz | raw");
    exp->add_option("--preset", ea.preset, "nuke (multi-layer EXR) | comfyui (PNG16 + NPZ) | rawdlss (binary + manifest)");
    exp->add_option("--range", ea.range, "frame range a-b (inclusive), a- or a");
    exp->add_option("--depth-dir", ea.depthDir, "depth pass folder for the nuke preset");
    exp->add_option("--mv-dir", ea.mvDir, "motion-vector pass folder for the nuke preset");
    exp->add_option("--matrix", ea.matrix, "colour matrix for YUV->RGB: bt601 | bt709 | auto")->default_val("auto");
    exp->add_option("--color-range", ea.colorRange, "limited | full | auto")->default_val("auto");
    exp->add_option("--pixel-type", ea.pixelType, "u8 | u16 | f16 | f32 (override)");
    exp->add_option("--scan-pass", ea.passName, "--from-dir without manifest: pass name");
    exp->add_option("--scan-convention", ea.convention, "--from-dir without manifest: raw | dlss");
    exp->add_option("--scan-size", ea.size, "--from-dir without manifest: WxH for raw dumps");

    ImportArgs ia;
    auto* imp = app.add_subcommand("import", "validate a pass folder and optionally convert it to / from the DLSS convention");
    AddImportOptions(imp, ia);
    imp->add_option("--expect-size", ia.expectSize, "required WxH");
    imp->add_option("--expect-frames", ia.expectFrames, "required frame count");
    imp->add_option("--to-dlss", ia.toDlss, "write depth_dlss / mv_dlss into this folder");
    imp->add_option("--to-raw", ia.toRaw, "write depth_raw / mv_raw into this folder");
    imp->add_flag("--write-manifest", ia.writeManifest, "write manifest.json for a folder that has none");

    ConvertArgs ca;
    auto* conv = app.add_subcommand("convert", "convert a pass between raw and DLSS conventions");
    AddImportOptions(conv, ca.in);
    conv->add_option("-o,--output", ca.output, "output folder")->required();
    conv->add_option("--to", ca.to, "depth_dlss | depth_raw | mv_dlss | mv_raw")->required();

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError& e) {
        return app.exit(e);
    }
    if (verbose) SetLogLevel(spdlog::level::debug);

    try {
        if (info->parsed()) return CmdInfo(infoInput, infoWarp);
        if (process->parsed()) return CmdProcess(pa);
        if (exp->parsed()) return CmdExport(ea);
        if (imp->parsed()) return CmdImport(ia);
        if (conv->parsed()) return CmdConvert(ca);
    } catch (const std::exception& e) {
        Log()->error("{}", e.what());
        return 1;
    }
    return 0;
}
