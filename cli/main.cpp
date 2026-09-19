// dlssvid — command line front-end. Everything the GUI can do must be reachable from here (ТЗ §7).
// Stage 0: `info`, `process --passthrough`. Stage 1: `export`, `import`, `convert`.

#include <CLI/CLI.hpp>

#include "FgCommands.h"
#include "ProcessCommands.h"
#include "NrCommands.h"
#include "UpscaleCommands.h"
#include "ViewportCommands.h"

#include <chrono>
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
#include "ml/ModelRegistry.h"
#include "stages/depth/DepthStage.h"
#include "stages/flow/FlowStage.h"
#include "util/Error.h"
#include "util/Log.h"
#include "util/Sha256.h"

#ifdef DLSSVID_WITH_CUDA
#include "gpu/CudaInterop.h"
#endif
#ifdef DLSSVID_WITH_TENSORRT
#include "ml/TrtLoader.h"
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
#ifdef DLSSVID_WITH_TENSORRT
    std::string trtReason;
    if (trt::Available(&trtReason)) std::printf("tensorrt:    %s (%s)\n", trt::LibraryVersion().c_str(), trt::LibraryDirectory().string().c_str());
    else std::printf("tensorrt:    unavailable (%s)\n", trtReason.c_str());
#else
    std::printf("tensorrt:    not built\n");
#endif
    return 0;
}

// ---- depth ----------------------------------------------------------------------------------

struct DepthArgs {
    std::string input, output;
    std::string backend = "da3";
    std::string model;
    int inputSize = 518;
    int maxRes = 1080;
    bool fp32 = false;
    int64_t frames = -1;
    bool noDlss = false;
    float zNear = 0.1f, zFar = 1000.f;
    std::string stabilize = "auto";  // auto | none | scale | scale_shift
    int stabilizeWindow = 8;
    bool noFill = false;
    std::string format = "exr";
    bool warp = false;
    std::string modelsDir;
    std::string python;
    std::string mvDir;
};

int CmdDepth(const DepthArgs& a) {
    VideoDecoder decoder(a.input);
    const auto& info = decoder.Info();
    DepthStageOptions o;
    o.backend = a.backend;
    o.estimator.modelId = a.model;
    o.estimator.inputSize = a.inputSize;
    o.estimator.maxInputRes = a.maxRes;
    o.estimator.fp16 = !a.fp32;
    o.estimator.modelsDir = a.modelsDir;
    if (!a.python.empty()) o.estimator.extra["python"] = a.python;
    o.outputDir = a.output;
    o.writeDlss = !a.noDlss;
    o.dlss.zNear = a.zNear;
    o.dlss.zFar = a.zFar;
    if (a.stabilize == "none") o.stabilize = TemporalStabilizer::Mode::None;
    else if (a.stabilize == "scale") o.stabilize = TemporalStabilizer::Mode::ScaleOnly;
    else if (a.stabilize == "scale_shift" || a.stabilize == "auto") o.stabilize = TemporalStabilizer::Mode::ScaleShift;
    else Throw("--stabilize must be auto|none|scale|scale_shift");
    o.stabilizeMetric = a.stabilize == "scale" || a.stabilize == "scale_shift";
    o.stabilizeWindow = a.stabilizeWindow;
    o.fillHoles = !a.noFill;
    const auto fmt = ParseFileFormat(a.format);
    if (!fmt || *fmt == FileFormat::Png) Throw("--format must be exr|tiff|npz|raw");
    o.format = *fmt;
    o.sourceFile = std::filesystem::path(a.input).filename().string();
    o.sourceHash = "sha256:" + Sha256File(a.input);
    o.mvDir = a.mvDir;
    const int64_t total = a.frames > 0 ? a.frames : info.frameCount;
    o.onFrame = [](int64_t frame, double tae) {
        if (frame % 10 == 0) std::fprintf(stderr, "  frame %lld TAE %.4f\n", static_cast<long long>(frame), tae);
    };

    D3D12Device device({a.warp, false});
    const auto t0 = std::chrono::steady_clock::now();
    const DepthRunResult r = RunDepth(decoder, device, o, a.frames, [total](int64_t n) { PrintProgress(n, total); });
    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::fprintf(stderr, "\n");
    Log()->info("depth done: {} frames in {:.1f} s ({:.1f} ms/frame), mean TAE {:.4f}, backend {}", r.stats.frames, sec,
                r.stats.frames ? 1000.0 * sec / r.stats.frames : 0.0, r.stats.MeanTae(), r.estimator.dump());
    std::printf("depth_raw:   %s\n", r.rawDir.string().c_str());
    if (!a.noDlss) std::printf("depth_dlss:  %s\n", r.dlssDir.string().c_str());
    std::printf("frames:      %lld\nmean TAE:    %.4f\nms/frame:    %.1f\n", static_cast<long long>(r.stats.frames), r.stats.MeanTae(),
                r.stats.frames ? 1000.0 * sec / r.stats.frames : 0.0);
    return 0;
}

// ---- flow -----------------------------------------------------------------------------------

struct FlowArgs {
    std::string input, output;
    std::string backend = "ofa";
    std::string model;
    std::string hwaccel = "cuda";
    int maxRes = 720;
    bool fp32 = false;
    int64_t frames = -1;
    bool noDlss = false;
    std::string target;
    int dilate = 1;
    std::string depthDir;
    std::string perf = "slow";
    int grid = 1;
    std::string format = "exr";
    bool warp = false;
    std::string modelsDir;
};

int CmdFlow(const FlowArgs& a) {
    VideoDecoder::Options dopt;
    if (a.hwaccel == "cuda") dopt.hwaccel = HwAccel::Cuda;
    else if (a.hwaccel != "none") Throw("--hwaccel must be none|cuda");
    VideoDecoder decoder(a.input, dopt);
    const auto& info = decoder.Info();
    FlowStageOptions o;
    o.backend = a.backend;
    o.estimator.modelId = a.model;
    o.estimator.maxRes = a.maxRes;
    o.estimator.fp16 = !a.fp32;
    o.estimator.modelsDir = a.modelsDir;
    o.estimator.extra["perf_level"] = a.perf;
    o.estimator.extra["grid"] = a.grid;
    o.outputDir = a.output;
    o.writeDlss = !a.noDlss;
    ParseSize(a.target, o.targetWidth, o.targetHeight);
    o.convert.dilateRadius = a.dilate;
    o.depthDir = a.depthDir;
    const auto fmt = ParseFileFormat(a.format);
    if (!fmt || *fmt == FileFormat::Png) Throw("--format must be exr|tiff|npz|raw");
    o.format = *fmt;
    o.sourceFile = std::filesystem::path(a.input).filename().string();
    o.sourceHash = "sha256:" + Sha256File(a.input);
    const int64_t total = a.frames > 0 ? a.frames : info.frameCount;
    o.onFrame = [](int64_t frame, double psnr) {
        if (frame % 10 == 0) std::fprintf(stderr, "  frame %lld warp PSNR %.2f dB\n", static_cast<long long>(frame), psnr);
    };
    D3D12Device device({a.warp, false});
    const auto t0 = std::chrono::steady_clock::now();
    const FlowRunResult r = RunFlow(decoder, device, o, a.frames, [total](int64_t n) { PrintProgress(n, total); });
    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::fprintf(stderr, "\n");
    Log()->info("flow done: {} frames in {:.1f} s ({:.1f} ms/frame), warp PSNR mean {:.2f} dB, backend {}", r.stats.frames, sec,
                r.stats.frames ? 1000.0 * sec / r.stats.frames : 0.0, r.stats.MeanWarpPsnr(), r.estimator.dump());
    std::printf("mv_raw:      %s\n", r.rawDir.string().c_str());
    if (!a.noDlss) std::printf("mv_dlss:     %s\n", r.dlssDir.string().c_str());
    std::printf("frames:      %lld\npairs:       %lld\nmean |v|:    %.2f px\nwarp PSNR:   %.2f dB (min %.2f)\nms/frame:    %.1f\n", static_cast<long long>(r.stats.frames),
                static_cast<long long>(r.stats.pairs), r.stats.meanMagnitude, r.stats.MeanWarpPsnr(), r.stats.psnrCount ? r.stats.minWarpPsnr : 0.0,
                r.stats.frames ? 1000.0 * sec / r.stats.frames : 0.0);
    return 0;
}

// ---- models ---------------------------------------------------------------------------------

int CmdModels(const std::string& action, const std::string& id, const std::string& modelsDir) {
    const auto registryPath = modelsDir.empty() ? ModelRegistry::DefaultRegistryPath() : std::filesystem::path(modelsDir) / "registry.json";
    const ModelRegistry reg = ModelRegistry::Load(registryPath);
    if (action == "list") {
        std::printf("registry:    %s\ncache:       %s\n", registryPath.string().c_str(), reg.CacheDir().string().c_str());
        for (const auto& e : reg.Entries()) {
            std::printf("%-20s %-6s %-10s %-14s %s%s\n", e.id.c_str(), e.stage.c_str(), e.format.c_str(), e.license.c_str(), e.role.c_str(),
                        reg.IsCached(e) ? "  [cached]" : "");
        }
        return 0;
    }
    if (action == "fetch") {
        const ModelEntry& e = reg.Get(id);
        const auto p = reg.Fetch(e, [](uint64_t done, uint64_t total) {
            if (total) std::fprintf(stderr, "\r%llu / %llu MiB", static_cast<unsigned long long>(done >> 20), static_cast<unsigned long long>(total >> 20));
        });
        std::fprintf(stderr, "\n");
        std::printf("%s -> %s\n", e.id.c_str(), p.string().c_str());
        return 0;
    }
    Throw("models: action must be list|fetch");
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

    cli::ProcessCommands processCommands;  // process (the whole pipeline), bench (stage 8)
    processCommands.Register(app);

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

    DepthArgs da;
    auto* depth = app.add_subcommand("depth", "estimate depth_raw / depth_dlss passes for a video");
    depth->add_option("-i,--input", da.input, "input video file")->required()->check(CLI::ExistingFile);
    depth->add_option("-o,--output", da.output, "pass root folder (writes depth_raw/ and depth_dlss/)")->required();
    depth->add_option("--backend", da.backend, "da3 | vda | worker | worker:da3 | worker:vda | worker:icdepth | stub")->default_val("da3");
    depth->add_option("--model", da.model, "registry model id (default per backend)");
    depth->add_option("--input-size", da.inputSize, "model input on the shorter side (multiple of 14)")->default_val(518);
    depth->add_option("--max-res", da.maxRes, "downscale frames above this shorter side before the model")->default_val(1080);
    depth->add_flag("--fp32", da.fp32, "build/run the model in fp32");
    depth->add_option("--frames", da.frames, "process at most N frames")->default_val(-1);
    depth->add_flag("--no-dlss", da.noDlss, "do not write depth_dlss");
    depth->add_option("--near", da.zNear, "near plane for depth_dlss (metres)")->default_val(0.1f);
    depth->add_option("--far", da.zFar, "far plane for depth_dlss (metres)")->default_val(1000.f);
    depth->add_option("--stabilize", da.stabilize, "temporal scale/shift: auto | none | scale | scale_shift")->default_val("auto");
    depth->add_option("--stabilize-window", da.stabilizeWindow, "frames in the alignment window")->default_val(8);
    depth->add_flag("--no-fill", da.noFill, "keep invalid depth samples instead of filling them");
    depth->add_option("--format", da.format, "exr | tiff | npz | raw")->default_val("exr");
    depth->add_flag("--warp", da.warp, "use the WARP software adapter");
    depth->add_option("--models-dir", da.modelsDir, "folder with registry.json (default: auto)");
    depth->add_option("--python", da.python, "python interpreter for the worker backends");
    depth->add_option("--mv-dir", da.mvDir, "mv_dlss pass folder: TAE with motion compensation");

    FlowArgs fa;
    auto* flow = app.add_subcommand("flow", "estimate motion vectors (mv_raw forward flow, mv_dlss backward) for a video");
    flow->add_option("-i,--input", fa.input, "input video file")->required()->check(CLI::ExistingFile);
    flow->add_option("-o,--output", fa.output, "pass root folder (writes mv_raw/ and mv_dlss/)")->required();
    flow->add_option("--backend", fa.backend, "ofa | searaft | stub")->default_val("ofa");
    flow->add_option("--model", fa.model, "registry model id (searaft)");
    flow->add_option("--hwaccel", fa.hwaccel, "decoder: cuda (NVDEC frames feed OFA directly) | none")->default_val("cuda");
    flow->add_option("--max-res", fa.maxRes, "searaft: model input cap on the shorter side")->default_val(720);
    flow->add_flag("--fp32", fa.fp32, "searaft: fp32 engine");
    flow->add_option("--frames", fa.frames, "process at most N frames")->default_val(-1);
    flow->add_flag("--no-dlss", fa.noDlss, "do not write mv_dlss");
    flow->add_option("--target", fa.target, "mv_dlss resolution WxH (default: source)");
    flow->add_option("--dilate", fa.dilate, "mv_dlss dilation radius at depth edges")->default_val(1);
    flow->add_option("--depth-dir", fa.depthDir, "depth_raw folder for occlusion handling in mv_dlss");
    flow->add_option("--perf", fa.perf, "ofa: slow | medium | fast")->default_val("slow");
    flow->add_option("--grid", fa.grid, "ofa: requested output grid (1, 2, 4)")->default_val(1);
    flow->add_option("--format", fa.format, "exr | tiff | npz | raw")->default_val("exr");
    flow->add_flag("--warp", fa.warp, "use the WARP software adapter");
    flow->add_option("--models-dir", fa.modelsDir, "folder with registry.json (default: auto)");

    cli::ViewportCommands viewportCommands;  // project init|show, render (stage 4)
    viewportCommands.Register(app);
    cli::UpscaleCommands upscaleCommands;  // upscale, compare (stage 5)
    upscaleCommands.Register(app);
    cli::NrCommands nrCommands;  // nr, nr-patch (stage 6)
    nrCommands.Register(app);
    cli::FgCommands fgCommands;  // fg (stage 7)
    fgCommands.Register(app);

    std::string modelsAction = "list", modelsId, modelsDir;
    auto* models = app.add_subcommand("models", "list or fetch models from models/registry.json");
    models->add_option("action", modelsAction, "list | fetch")->default_val("list");
    models->add_option("id", modelsId, "model id for fetch");
    models->add_option("--models-dir", modelsDir, "folder with registry.json (default: auto)");

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError& e) {
        return app.exit(e);
    }
    if (verbose) SetLogLevel(spdlog::level::debug);

    try {
        if (info->parsed()) return CmdInfo(infoInput, infoWarp);
        if (const int rc = processCommands.Dispatch(); rc >= 0) return rc;
        if (exp->parsed()) return CmdExport(ea);
        if (imp->parsed()) return CmdImport(ia);
        if (conv->parsed()) return CmdConvert(ca);
        if (depth->parsed()) return CmdDepth(da);
        if (flow->parsed()) return CmdFlow(fa);
        if (models->parsed()) return CmdModels(modelsAction, modelsId, modelsDir);
        if (const int rc = viewportCommands.Dispatch(); rc >= 0) return rc;
        if (const int rc = upscaleCommands.Dispatch(); rc >= 0) return rc;
        if (const int rc = nrCommands.Dispatch(); rc >= 0) return rc;
        if (const int rc = fgCommands.Dispatch(); rc >= 0) return rc;
    } catch (const std::exception& e) {
        Log()->error("{}", e.what());
        return 1;
    }
    return 0;
}
