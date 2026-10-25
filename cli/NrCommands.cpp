#include "NrCommands.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <vector>

#include "gpu/D3D12Device.h"
#include "io/VideoDecoder.h"
#include "passes/Manifest.h"
#include "stages/nr/INrBackend.h"
#include "stages/nr/NrPatch.h"
#include "stages/ParamSchema.h"
#include "stages/nr/NrStage.h"
#include "util/Error.h"
#include "util/Log.h"
#include "util/Sha256.h"

namespace dlssvid::cli {

namespace {

void Progress(int64_t done, int64_t total) {
    if (done % 10 == 0 || done == total) {
        if (total > 0) std::fprintf(stderr, "\r%lld/%lld frames", static_cast<long long>(done), static_cast<long long>(total));
        else std::fprintf(stderr, "\r%lld frames", static_cast<long long>(done));
    }
}

void ParseSize(const std::string& s, uint32_t& w, uint32_t& h) {
    const size_t x = s.find('x');
    if (x == std::string::npos) Throw("--check-size expects WxH (got '" + s + "')");
    w = static_cast<uint32_t>(std::stoul(s.substr(0, x)));
    h = static_cast<uint32_t>(std::stoul(s.substr(x + 1)));
}

std::string MiB(int64_t b) {
    if (b < 0) return "?";
    char s[32];
    std::snprintf(s, sizeof(s), "%.1f MiB", static_cast<double>(b) / 1048576.0);
    return s;
}

std::string Ms(double v) {
    if (v < 0) return "?";
    char s[32];
    std::snprintf(s, sizeof(s), "%.0f ms", v);
    return s;
}

void PrintDiagnostics(const NrDiagnostics& d, const std::string& error, double deviceMs) {
    std::printf("GPU:          %s\n", d.gpu.c_str());
    std::printf("architecture: %s\n", d.architecture.empty() ? "?" : d.architecture.c_str());
    std::printf("driver:       %s%s\n", d.driver.ToString().c_str(), !d.driver.valid ? "" : d.driverOk ? " (>= 616.56)" : " (< 616.56 required)");
    if (d.dll.empty()) std::printf("dll:          not found (%s)\n", kNrDllName);
    else std::printf("dll:          %s (%llu bytes, sha256 %s)\n", d.dll.string().c_str(), static_cast<unsigned long long>(d.dllSize), d.dllSha256.c_str());
    std::printf("forwarder:    %s\n", d.forwarder.empty() ? "not found" : d.forwarder.string().c_str());
    if (!d.initResult.empty()) std::printf("init:         %s%s\n", d.initResult.c_str(), d.initAbi >= 0 ? (" (abi " + std::to_string(d.initAbi) + ")").c_str() : "");
    if (!d.createResult.empty()) std::printf("create(18):   %s [%s block]\n", d.createResult.c_str(), d.paramsBlock.c_str());
    if (d.initMs >= 0) {  // the backend reached its init: the cost of creating the feature (spec E8, E6)
        std::printf("time:         device %s, init %s, create %s\n", Ms(deviceMs).c_str(), Ms(d.initMs).c_str(), Ms(d.createMs).c_str());
        // «unknown» only when DXGI gave no reading at all: a CreateFeature that failed leaves the feature unmeasured, not the
        // adapter (review of TASK-0048: the unpatched DLL on RTX 40 read as «the adapter does not report it»)
        if (d.vramBudgetBytes < 0 && d.vramInitBytes < 0 && d.vramFeatureBytes < 0) std::printf("vram:         unknown (the adapter does not report it)\n");
        else
            std::printf("vram:         feature %s%s, init %s, budget %s (DXGI, this process)\n", MiB(d.vramFeatureBytes).c_str(),
                        d.vramFeatureBytes < 0 ? " (not created)" : "", MiB(d.vramInitBytes).c_str(), MiB(d.vramBudgetBytes).c_str());
    }
    if (d.ok) std::printf("status:       ok\n");
    else std::printf("status:       FAILED: %s\n", !d.hint.empty() ? d.hint.c_str() : error.c_str());
}

std::string Join(const std::vector<std::string>& items, const char* sep = "; ") {
    std::string all;
    for (const auto& s : items) all += (all.empty() ? "" : sep) + s;
    return all;
}

// Every option of an nr setting as the schema spells it, defaults included; ResolveNrParams keeps the given ones.
nlohmann::json ArgsAsParams(const NrCommands::NrArgs& a) {
    NrStageOptions o;
    o.backend = a.backend;
    o.intensity = a.intensity;
    o.style = a.style;
    o.localTone = a.localTone;
    o.localStructure = a.localStructure;
    o.skinStructure = a.skinStructure;
    o.autoMask = a.autoMask;
    o.passes = a.passes;
    o.modelScale = a.modelScale;
    o.transfer = a.transfer;
    o.maxRatio = a.maxRatio;
    o.useGuides = !a.noGuides;
    o.temporal = a.temporal;
    o.temporalThreshold = a.temporalThreshold;
    o.skinBlend = a.skinBlend;
    o.tonemap.curve = a.tonemap;
    o.tonemap.inputTransfer = a.inputTransfer;
    o.tonemap.exposureEv = a.exposure;
    nlohmann::json p = NrParamsFromOptions(o);  // the style by its schema name, guides from --no-guides
    const auto fmt = ParseFileFormat(a.format);
    if (!fmt) Throw("--format must be exr | png16 | tiff | npz | raw");  // what the schema offers (TASK-0044: exr | png16 only)
    p["format"] = *fmt == FileFormat::Png ? std::string("png") : std::string(ToString(*fmt));  // the schema's choice: ToString says png16
    return p;
}

// The run's nr settings (spec §2h): the schema's defaults < --params-file < the options given on the command line. «Given» is
// CLI::Option::count() > 0, not «differs from the default»: `--temporal 0` over a file with 0.4 turns the filter off. The keys
// come from the schema, so a new key cannot be forgotten here (TASK-0047).
nlohmann::json ResolveNrParams(const NrCommands::NrArgs& a, const CLI::App& cmd) {
    nlohmann::json params = nlohmann::json::object();
    if (!a.paramsFile.empty()) {
        std::vector<std::string> dropped;
        params = ReadStageParamsFile("nr", a.paramsFile, &dropped);
        // refused, not skipped: a typo, or an option of the run written into the file, would change nothing and say nothing -
        // an A/B variant would compare two equal runs. An option is recognised by the parser itself (every long option of nr
        // has its snake spelling), so the message never names an option that does not exist (review of TASK-0047)
        std::vector<std::string> problems;
        for (std::string k : dropped) {
            std::replace(k.begin(), k.end(), '-', '_');
            const bool isOption = !k.empty() && cmd.get_option_no_throw("--" + k) != nullptr;
            problems.push_back(isOption ? "'" + k + "' is an option of dlssvid nr, not a setting: give it on the command line"
                                        : "'" + k + "' is not an nr setting (stages.nr.params keys)");
        }
        if (!problems.empty()) Throw("nr: --params-file " + a.paramsFile + ": " + Join(problems));
    }
    const nlohmann::json all = ArgsAsParams(a);
    for (const ParamSpec& p : FindStageSchema("nr")->params) {
        // each key has an option under its snake spelling (test_cli: every schema key in --help), guides as --no_guides / --guides
        const CLI::Option* opt = cmd.get_option_no_throw(p.key == "guides" ? "--no_guides" : "--" + p.key);
        if (!opt) Throw("nr: internal: the setting '" + p.key + "' has no option");
        // a number as typed (a float member widened 0.3 to 0.30000001192...; the file's 0.3 stays 0.3 - one value in --json)
        if (opt->count() > 0) params[p.key] = p.type == ParamType::Float ? nlohmann::json(opt->as<double>()) : all.at(p.key);
    }
    // the file's values and the flags by one schema, before anything opens the GPU (a wrong JSON type: a message, not type_error)
    if (const auto problems = ValidateStageParams("nr", params); !problems.empty()) Throw("nr: " + Join(problems));
    return params;
}

// --start F --frames N --warmup K (spec §2f): decode from F-K so the model (Reset there) and the temporal filter build their
// history, write F..F+N-1. Indices are those of the pass folders and the viewport. SeekToFrame maps them to pts at the
// stream's rate, i.e. assumes a constant frame rate: VFR is not detected yet (matrix media-vfr; PR-10 refuses it).
struct NrRange {
    int64_t first = 0, warmup = 0, from = 0, maxFrames = -1;  // maxFrames: frames to decode, -1 to the end
    bool Ranged() const { return first > 0; }
};

// The frames a pass folder has against the frames the run reads: a missing colour frame stops the run after the model's
// start-up, a missing guide frame silently drops that guide - an A/B would compare different inputs.
void CheckCoverage(const std::filesystem::path& dir, const char* what, int64_t from, int64_t last, bool required) {
    if (dir.empty() || !Manifest::Exists(dir)) return;
    const Manifest m = Manifest::Load(dir);
    if (m.firstFrame <= from && (last < 0 || m.lastFrame >= last)) return;
    const std::string text = std::string(what) + " " + dir.string() + " has frames " + std::to_string(m.firstFrame) + ".." + std::to_string(m.lastFrame) +
                             ", the run reads " + std::to_string(from) + ".." + (last < 0 ? std::string("the end") : std::to_string(last));
    if (required) Throw("nr: " + text);
    Log()->warn("nr: {}; frames outside go without it", text);
}

// A range or an A/B run replaces color_nr with a pass process never takes as the project's: never over the pass of a processed
// project (it has a fingerprint) - written in place, it would be lost; such runs go under their own -o with --color-dir /
// --depth-dir / --mv-dir / --masks-dir (TASK-0047; A/B runs: review of TASK-0048)
void GuardProcessedPass(const NrStageOptions& o, const char* what) {
    const auto out = o.outputDir / "color_nr";
    if (!Manifest::Exists(out)) return;
    std::string fp;
    try {
        fp = Manifest::Load(out).fingerprint;
    } catch (const std::exception&) {
    }
    if (!fp.empty())
        Throw("nr: " + out.string() + " is the pass of a processed project; write " + what + " under another -o (inputs with --color-dir / --depth-dir / " +
              "--mv-dir / --masks-dir)");
}

NrRange PrepareRange(const NrCommands::NrArgs& a, VideoDecoder& decoder, NrStageOptions& o) {
    NrRange r;
    r.first = a.start;
    r.warmup = std::min(a.warmup, a.start);  // nothing before the clip's first frame
    if (r.warmup < a.warmup) Log()->info("nr: --warmup {} shortened to {}: the clip starts at frame 0", a.warmup, r.warmup);
    r.from = r.first - r.warmup;
    r.maxFrames = a.frames >= 0 ? r.warmup + a.frames : -1;
    o.writeFrom = r.first;
    if (!r.Ranged()) return r;
    if (a.frames == 0) Throw("nr: --frames 0 with --start " + std::to_string(r.first) + " writes nothing: give N >= 1 or leave --frames out");
    const int64_t clipFrames = decoder.Info().frameCount;  // 0: unknown (mkv)
    if (clipFrames > 0 && r.first >= clipFrames)
        Throw("nr: --start " + std::to_string(r.first) + " is past the end of the clip (" + std::to_string(clipFrames) + " frames)");
    GuardProcessedPass(o, "the range");
    // the last frame the run reads: --frames is «at most N», the clip may end first; with the clip's length unknown (mkv) only
    // the start of a pass is checked - a short colour pass still stops the run at «has no frame N» (review of TASK-0047)
    int64_t last = a.frames > 0 ? r.first + a.frames - 1 : -1;
    if (clipFrames > 0) last = last < 0 ? clipFrames - 1 : std::min(last, clipFrames - 1);
    else last = -1;
    CheckCoverage(o.colorDir, "the colour pass", r.from, last, true);
    if (o.useGuides) {
        CheckCoverage(o.depthDir, "the depth pass", r.from, last, false);
        CheckCoverage(o.mvDir, "the motion-vector pass", r.from, last, false);
    }
    if (r.from > 0) {
        Log()->info("nr: frames from {} through the model (warm-up {}), written from {}; the seek assumes a constant frame rate", r.from, r.warmup, r.first);
        if (!decoder.SeekToFrame(r.from)) Throw("nr: --start " + std::to_string(r.first) + " is past the end of " + a.input);
    }
    return r;
}

int CmdNrCheck(const NrCommands::NrArgs& a, const NrStageOptions& o) {
    // the values were checked with the run's (ResolveNrParams); the feature is created with the run's tuning at --check-size
    const auto t0 = std::chrono::steady_clock::now();
    D3D12Device device({a.warp, false});
    const double deviceMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::unique_ptr<INrBackend> backend = CreateNrBackend(o.backend);
    uint32_t w = 0, h = 0;
    ParseSize(a.checkSize, w, h);
    NrConfig cfg = NrConfigFromOptions(o, w, h);
    if (cfg.createWithMv) {  // no pass folders here: zero guides at the check size (E4: the create time and VRAM with guides bound)
        cfg.createMvWidth = w;
        cfg.createMvHeight = h;
        if (cfg.depthGuide) {
            cfg.createDepthWidth = w;
            cfg.createDepthHeight = h;
        }
    }
    std::string error;
    try {
        backend->Init(device, cfg);
    } catch (const std::exception& e) {
        error = e.what();
    }
    const NrDiagnostics d = backend->Diagnostics();
    PrintDiagnostics(d, error, deviceMs);
    auto usage = [&]() -> int64_t {
        const auto m = device.LocalVideoMemory();
        return m ? static_cast<int64_t>(m->currentUsage) : -1;
    };
    const int64_t usageAfterInit = d.ok ? usage() : -1;
    bool failed = !d.ok;

    // E6: more features alive with the first - each a whole Init as the passes of a run are - until one fails (the limit observed) or
    // the process nears 80 % of its DXGI budget (a TDR or an out-of-memory would take the desktop with it)
    nlohmann::json features = nlohmann::json::array();
    std::string featuresStop;
    std::vector<std::unique_ptr<INrBackend>> more;
    if (d.ok && a.checkFeatures > 0) {
        auto entry = [&](int index, const NrDiagnostics& fd, const std::string& err) {
            features.push_back({{"index", index},
                                {"ok", fd.ok},
                                {"create_result", fd.createResult},
                                {"init_ms", fd.initMs < 0 ? nlohmann::json(nullptr) : nlohmann::json(fd.initMs)},
                                {"create_ms", fd.createMs < 0 ? nlohmann::json(nullptr) : nlohmann::json(fd.createMs)},
                                {"vram_feature_bytes", fd.vramFeatureBytes < 0 ? nlohmann::json(nullptr) : nlohmann::json(fd.vramFeatureBytes)},
                                {"vram_usage_bytes", usage() < 0 ? nlohmann::json(nullptr) : nlohmann::json(usage())},
                                {"error", err}});
            std::printf("feature %2d:   %s, init %s, create %s, feature %s, process %s\n", index, fd.ok ? "ok" : "FAILED", Ms(fd.initMs).c_str(),
                        Ms(fd.createMs).c_str(), MiB(fd.vramFeatureBytes).c_str(), MiB(usage()).c_str());
        };
        entry(1, d, "");
        for (int i = 2; i <= a.checkFeatures; ++i) {
            const auto m = device.LocalVideoMemory();
            if (m && m->budget > 0 && static_cast<double>(m->currentUsage) > 0.8 * static_cast<double>(m->budget)) {
                featuresStop = "budget";
                std::printf("features:     stopped before %d: %s of the %s budget in use (80 %% guard)\n", i, MiB(static_cast<int64_t>(m->currentUsage)).c_str(),
                            MiB(static_cast<int64_t>(m->budget)).c_str());
                break;
            }
            std::unique_ptr<INrBackend> b = CreateNrBackend(o.backend);
            std::string err;
            try {
                b->Init(device, cfg);
            } catch (const std::exception& e) {
                err = e.what();
            }
            entry(i, b->Diagnostics(), err);
            if (!err.empty()) {
                featuresStop = "failure";
                failed = true;
                b->Shutdown();
                break;
            }
            more.push_back(std::move(b));
        }
        std::printf("features:     %zu alive at once%s\n", more.size() + 1, featuresStop.empty() ? "" : (" (" + featuresStop + ")").c_str());
    }

    // E8: re-creations inside the same Init (the preview's phase 2 re-creates the feature on a change of its settings); the first
    // failure stops - it is the answer
    // stops: a failure (the answer), the 80 % budget guard of E6, or memory growing by more than one feature - a release that frees
    // nothing would stack a feature per round, and many features hung the machine (INCIDENT 2026-10-25)
    nlohmann::json recreations = nlohmann::json::array();
    std::string recreateStop;
    if (d.ok && a.checkRepeat > 0) {
        int okCount = 0;
        for (int i = 1; i <= a.checkRepeat; ++i) {
            const auto m = device.LocalVideoMemory();
            if (m && m->budget > 0 && static_cast<double>(m->currentUsage) > 0.8 * static_cast<double>(m->budget)) {
                recreateStop = "budget";
                failed = true;
                break;
            }
            NrRecreation r;
            try {
                r = backend->RecreateFeature();
            } catch (const std::exception& e) {  // a removed device or a failed submit: the rounds so far and the JSON are kept
                r.ok = false;
                r.result = std::string("exception: ") + e.what();
            }
            nlohmann::json e = r.ToJson();
            e["index"] = i;
            recreations.push_back(std::move(e));
            std::printf("recreate %3d: %s, release %s, create %s, process %s after release, %s after create\n", i, r.result.c_str(), Ms(r.releaseMs).c_str(),
                        Ms(r.createMs).c_str(), MiB(r.vramAfterRelease).c_str(), MiB(r.vramAfterCreate).c_str());
            if (!r.ok) {
                recreateStop = "failure";
                failed = true;
                break;
            }
            ++okCount;
            if (usageAfterInit >= 0 && r.vramAfterCreate >= 0 && d.vramFeatureBytes > 0 && r.vramAfterCreate - usageAfterInit > d.vramFeatureBytes) {
                recreateStop = "growth";
                failed = true;
                break;
            }
        }
        std::printf("recreate:     %d of %d ok (process %s after the first Init)%s\n", okCount, a.checkRepeat, MiB(usageAfterInit).c_str(),
                    recreateStop.empty() ? "" : (", stopped: " + recreateStop).c_str());
    }
    for (auto it = more.rbegin(); it != more.rend(); ++it) (*it)->Shutdown();
    backend->Shutdown();
    if (!a.json.empty()) {
        nlohmann::json j = d.ToJson();
        j["error"] = error;
        j["device_ms"] = deviceMs;
        j["ab"] = NrAbJson(o);
        j["vram_usage_after_init_bytes"] = usageAfterInit < 0 ? nlohmann::json(nullptr) : nlohmann::json(usageAfterInit);
        if (a.checkFeatures > 0) {
            j["features"] = features;
            j["features_alive"] = d.ok ? static_cast<int>(more.size()) + 1 : 0;
            j["features_stop"] = featuresStop;
        }
        if (a.checkRepeat > 0) {
            j["recreations"] = recreations;
            j["recreations_stop"] = recreateStop;
        }
        std::ofstream(a.json) << j.dump(2) << "\n";
    }
    return failed ? 1 : 0;
}

int CmdNr(const NrCommands::NrArgs& a, const CLI::App& cmd) {
    const nlohmann::json params = ResolveNrParams(a, cmd);
    NrStageOptions o;
    ApplyNrParams(params, o);  // keys not given keep the options' defaults - the schema's (test_param_schema)
    o.paramsBlock = a.paramsBlock;
    o.requireDriver = !a.skipDriverCheck;
    o.dllDir = a.dllDir;
    o.mvScale = a.mvScale;  // the A/B switches: options of the run, never settings (spec §2e)
    o.createWithMv = a.createWithMv;
    o.depthGuide = !a.noDepthGuide;
    if (const auto problems = NrAbProblems(o); !problems.empty()) Throw("nr: " + Join(problems));
    if (a.checkRepeat > 0 && a.checkFeatures > 0)
        Throw("nr: --check-repeat and --check-features measure one thing each - give one of them per run");
    if ((a.checkRepeat > 0 || a.checkFeatures > 0) && !a.check) Throw("nr: --check-repeat and --check-features belong to --check");
    if (a.check) return CmdNrCheck(a, o);
    if (a.input.empty()) Throw("nr: -i/--input is required (or --check)");
    if (a.output.empty()) Throw("nr: -o/--output is required");
    VideoDecoder decoder(a.input);
    const auto& info = decoder.Info();
    // pass folders produced by the earlier stages live under the same root: pick them up unless told otherwise
    auto autoDir = [&](const std::string& given, const char* name) -> std::filesystem::path {
        if (!given.empty()) return given;
        if (a.noAuto) return {};
        const auto d = std::filesystem::path(a.output) / name;
        if (std::filesystem::exists(d / Manifest::kFileName)) {
            Log()->info("nr: using {}", d.string());
            return d;
        }
        return {};
    };
    o.colorDir = autoDir(a.colorDir, "color_sr");
    o.depthDir = autoDir(a.depthDir, "depth_dlss");
    o.mvDir = autoDir(a.mvDir, "mv_dlss");
    o.masksDir = !a.masksDir.empty() ? std::filesystem::path(a.masksDir) : a.noAuto ? std::filesystem::path() : std::filesystem::path(a.output);
    o.outputDir = a.output;
    o.format = *ParseFileFormat(params.value("format", std::string("exr")));  // validated by the schema
    o.videoOut = a.video;
    o.videoCodec = a.codec;
    o.sourceFile = std::filesystem::path(a.input).filename().string();
    o.sourceHash = "sha256:" + Sha256File(a.input);
    const NrRange range = PrepareRange(a, decoder, o);
    if (!range.Ranged() && !NrAbJson(o).empty()) GuardProcessedPass(o, "an A/B run");
    const int64_t total = range.maxFrames >= 0 ? range.maxFrames : info.frameCount > 0 ? info.frameCount - range.from : 0;
    o.onFrame = [](int64_t frame, double ms) {
        if (frame % 10 == 0) std::fprintf(stderr, "  frame %lld %.1f ms\n", static_cast<long long>(frame), ms);
    };
    D3D12Device device({a.warp, false});
    const auto t0 = std::chrono::steady_clock::now();
    const NrRunResult r = RunNr(decoder, device, o, range.maxFrames, [total](int64_t n) { Progress(n, total); });
    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::fprintf(stderr, "\n");
    if (r.stats.disabled) {
        std::printf("nr:          disabled — %s\n", r.stats.disabledReason.c_str());
        return 1;
    }
    if (range.Ranged() && r.stats.written == 0) Throw("nr: nothing written - --start " + std::to_string(range.first) + " is past the end of " + a.input);
    if (a.frames > 0 && r.stats.written < a.frames)
        Log()->warn("nr: the clip ends at frame {}: {} of {} frames written", r.stats.lastWritten, r.stats.written, a.frames);
    Log()->info("nr done: {} frames in {:.1f} s ({:.1f} ms/frame wall, {:.1f} ms/frame GPU), backend {}", r.stats.frames, sec,
                r.stats.frames ? 1000.0 * sec / r.stats.frames : 0.0, r.stats.MeanMs(), r.backend.dump());
    std::printf("color_nr:    %s\n", r.outDir.string().c_str());
    std::printf("backend:     %s\n", r.stats.backend.c_str());
    std::printf("colour from: %s\n", r.stats.colorSource.c_str());
    std::printf("size:        %ux%u (model at %ux%u)\n", r.stats.width, r.stats.height, r.stats.workWidth, r.stats.workHeight);
    std::printf("passes:      %d\nguides:      depth %s, mv %s\n", r.stats.passes, r.stats.depthUsed ? "yes" : "no", r.stats.mvUsed ? "yes" : "no");
    std::string masks;
    for (const auto& m : r.stats.masks) masks += (masks.empty() ? "" : ", ") + m;
    std::printf("masks:       %s\n", masks.empty() ? "none" : masks.c_str());
    std::printf("frames:      %lld\nms/frame:    %.1f (GPU %.1f)\n", static_cast<long long>(r.stats.frames), r.stats.frames ? 1000.0 * sec / r.stats.frames : 0.0,
                r.stats.MeanMs());
    if (range.Ranged())
        std::printf("range:       %lld..%lld (warm-up %lld)\n", static_cast<long long>(r.stats.firstWritten), static_cast<long long>(r.stats.lastWritten),
                    static_cast<long long>(r.stats.warmup));
    if (r.stats.mvScaleX > 0) std::printf("mv scale:    %.3f x %.3f (%s)\n", r.stats.mvScaleX, r.stats.mvScaleY, a.mvScale.c_str());
    if (const auto ab = NrAbJson(o); !ab.empty()) std::printf("a/b:         %s (the pass carries stage_params.ab: process never adopts it)\n", ab.dump().c_str());
    if (!a.video.empty()) std::printf("video:       %s\n", a.video.c_str());
    if (!a.json.empty()) {
        nlohmann::json j = {{"color_nr", r.outDir.string()},
                            {"backend", r.backend},
                            {"diagnostics", r.diagnostics},
                            {"params", EffectiveStageParams("nr", params)},  // the run's settings: defaults < --params-file < the options given
                            {"ab", NrAbJson(o)},
                            {"stats",
                             {{"frames", r.stats.frames},  // through the model, warm-up included
                              {"frames_written", r.stats.written},
                              {"warmup_frames", r.stats.warmup},
                              {"ms_per_frame_gpu", r.stats.MeanMs()},
                              {"ms_per_frame_wall", r.stats.frames ? 1000.0 * sec / r.stats.frames : 0.0},
                              {"size", {r.stats.width, r.stats.height}},
                              {"work_size", {r.stats.workWidth, r.stats.workHeight}},
                              {"passes", r.stats.passes},
                              {"depth_used", r.stats.depthUsed},
                              {"mv_used", r.stats.mvUsed},
                              {"masks", r.stats.masks},
                              {"color_source", r.stats.colorSource},
                              {"mv_scale", r.stats.mvScaleX > 0 ? nlohmann::json::array({r.stats.mvScaleX, r.stats.mvScaleY}) : nlohmann::json(nullptr)}}}};
        if (range.Ranged()) j["range"] = {{"first", r.stats.firstWritten}, {"last", r.stats.lastWritten}, {"warmup", r.stats.warmup}};
        std::ofstream(a.json) << j.dump(2) << "\n";
    }
    return 0;
}

int CmdPatch(const NrCommands::PatchArgs& a) {
    NrPatchOptions o;
    o.input = a.input;
    o.output = a.output;
    o.patcher = a.patcher;
    o.cudaBin = a.cudaBin;
    o.python = a.python;
    o.dryRun = a.dryRun;
    if (a.arch != "all") {
        std::stringstream ss(a.arch);
        std::string part;
        while (std::getline(ss, part, ','))
            if (!part.empty()) o.archs.push_back(part);
    }
    const NrPatchResult r = RunNrPatch(o);
    std::printf("patcher:     %s\n", r.script.string().c_str());
    std::printf("input:       %s (sha256 %s)\n", a.input.c_str(), r.inputSha256.c_str());
    if (r.ok && !a.dryRun) {
        std::printf("output:      %s (sha256 %s)\n", r.output.string().c_str(), r.outputSha256.c_str());
        std::printf("status:      ok — run `dlssvid nr --check` to verify CreateFeature(18)\n");
    } else if (r.ok) {
        std::printf("status:      dry run ok\n");
    } else {
        std::printf("status:      FAILED (patcher exit code %u) — see the patcher output above\n", r.exitCode);
    }
    return r.ok ? 0 : 1;
}

}  // namespace

void NrCommands::Register(CLI::App& app) {
    nr_ = app.add_subcommand("nr", "DLSS 5 Neural Rendering (NGX Feature 18) -> color_nr; --check prints the GPU / driver / DLL diagnostics only");
    nr_->add_option("-i,--input", na_.input, "input video file")->check(CLI::ExistingFile);
    nr_->add_option("-o,--output", na_.output, "pass root folder (writes color_nr/; color_sr, depth_dlss, mv_dlss and mask_* found here are used)");
    nr_->add_option("--color-dir,--color_dir", na_.colorDir, "colour input pass folder (default: <output>/color_sr when present, else the video)");
    nr_->add_option("--depth-dir,--depth_dir", na_.depthDir, "depth_dlss pass folder (guide)");
    nr_->add_option("--mv-dir,--mv_dir", na_.mvDir, "mv_dlss pass folder (guide)");
    nr_->add_option("--masks-dir,--masks_dir", na_.masksDir, "root with mask_ui / mask_ignore / mask_face / mask_skin pass folders (default: <output>)");
    nr_->add_flag("--no-auto,--no_auto", na_.noAuto, "do not pick pass folders under --output automatically");
    nr_->add_option("--backend", na_.backend, "ngx (nvngx_dlssnr.dll) | stub (deterministic test edit)")->default_val("ngx");
    nr_->add_option("--intensity", na_.intensity, "DLSSNR.Intensity 0..2")->default_val(1.f);
    nr_->add_option("--style", na_.style, "natural | cinematic | default")->default_val("natural");
    nr_->add_option("--local-tone,--local_tone", na_.localTone, "DLSSNR.LocalToneStrength 0..2")->default_val(1.f);
    nr_->add_option("--local-structure,--local_structure", na_.localStructure, "DLSSNR.LocalStructureStrength 0..2")->default_val(1.f);
    nr_->add_option("--skin-structure,--skin_structure", na_.skinStructure, "DLSSNR.SkinStructureStrength (-1 = model default)")->default_val(-1.f);
    nr_->add_flag("--auto-mask,--auto_mask,!--no-auto-mask,!--no_auto_mask", na_.autoMask,
                  "DLSSNR.UseAutoMask (the model's automatic skin mask); --no-auto-mask overrides --params-file");
    nr_->add_option("--passes", na_.passes, "model passes 1 | 2")->default_val(1);
    nr_->add_option("--model-scale,--model_scale", na_.modelScale, "model resolution relative to the colour (0.25..2); != 1 uses the ratio transfer resolve")->default_val(1.0);
    nr_->add_option("--transfer", na_.transfer, "ratio transfer strength 0..1 (model-scale != 1)")->default_val(1.f);
    nr_->add_option("--max-ratio,--max_ratio", na_.maxRatio, "ratio transfer guard")->default_val(4.f);
    nr_->add_flag("--no-guides,--no_guides,!--guides", na_.noGuides, "A/B: no depth / motion vectors to the model (still-image mode); --guides overrides --params-file");
    // A/B switches (spec §2e, E3-E5): standalone only, not schema keys - a project, a preset, --params-file or `process` never sets
    // them, and a pass made with them (stage_params.ab) is never adopted by `process`; PR-7 wires the winner in and drops them
    nr_->add_option("--mv-scale,--mv_scale", na_.mvScale, "A/B: DLSSNR.MVecScale - unit (mv_dlss pixels as is, the sub-rect carries the size) | auto (x model size / mv size)")
        ->default_val("unit");
    nr_->add_flag("--create-with-mv,--create_with_mv", na_.createWithMv, "A/B: CreateFeature with zero motion vectors (and depth) bound instead of none (ComfyUI)");
    nr_->add_flag("--no-depth-guide,--no_depth_guide", na_.noDepthGuide, "A/B: motion vectors to the model, depth not");
    nr_->add_option("--temporal", na_.temporal, "temporal filter weight 0..1 (0 = off): blend with the previous frame warped by mv_dlss")->default_val(0.f);
    nr_->add_option("--temporal-threshold,--temporal_threshold", na_.temporalThreshold, "colour difference that rejects the temporal blend")->default_val(0.1f);
    nr_->add_option("--skin-blend,--skin_blend", na_.skinBlend, "edit strength inside mask_face / mask_skin (0 = keep the input)")->default_val(1.f);
    nr_->add_option("--tonemap", na_.tonemap, "passthrough | aces | reinhard")->default_val("passthrough");
    nr_->add_option("--input-transfer,--input_transfer", na_.inputTransfer, "srgb | linear | pq | hlg")->default_val("srgb");
    nr_->add_option("--exposure", na_.exposure, "exposure in stops (EV) before the tonemap: +1 doubles linear light, 0 = unchanged")->default_val(0.f);
    nr_->add_option("--format", na_.format, "exr | png16 | tiff | npz | raw")->default_val("exr");
    nr_->add_option("--video", na_.video, "also encode the result into this video file (no audio)");
    nr_->add_option("--codec", na_.codec, "encoder for --video: h264_nvenc | hevc_nvenc | av1_nvenc | ffv1")->default_val("h264_nvenc");
    nr_->add_option("--frames", na_.frames, "write at most N frames (from --start; warm-up frames not counted)")->default_val(-1);
    nr_->add_option("--start", na_.start, "first frame to write, as the pass folders and the viewport count (from 0); the seek assumes a constant frame rate")
        ->default_val(0)
        ->check(CLI::NonNegativeNumber);
    nr_->add_option("--warmup", na_.warmup, "frames before --start that go through the model for its history and are not written (fewer at the clip's start)")
        ->default_val(0)
        ->check(CLI::NonNegativeNumber);
    nr_->add_option("--params-file,--params_file", na_.paramsFile,
                    "JSON object of nr settings under the project's keys (stages.nr.params); options given on the command line win")
        ->check(CLI::ExistingFile);
    nr_->add_flag("--warp", na_.warp, "use the WARP software adapter (stub)");
    nr_->add_option("--dll-dir,--dll_dir", na_.dllDir, "folder with nvngx_dlssnr.dll (default: bin/nvidia next to the executable)");
    nr_->add_option("--params-block,--params_block", na_.paramsBlock, "NGX parameter block: capability | alloc")->default_val("capability");
    nr_->add_flag("--skip-driver-check,--skip_driver_check", na_.skipDriverCheck, "try even with a driver older than 616.56");
    nr_->add_flag("--check", na_.check, "no processing: initialise the backend, print GPU / driver / DLL / CreateFeature(18) diagnostics and exit 0/1");
    nr_->add_option("--check-size,--check_size", na_.checkSize, "resolution for --check")->default_val("1280x720");
    nr_->add_option("--check-repeat,--check_repeat", na_.checkRepeat,
                    "--check: release and create the feature N more times inside one init (times, VRAM after each; the first failure stops)")
        ->check(CLI::Range(0, 1000));
    // at most 4 (passes 1..4, spec §2c): a probe of up to 18 NGX features at 1440p hung the whole machine with no TDR and no bugcheck,
    // a hard reset was needed (INCIDENT 2026-10-25, TASK-0053) - the 80 % budget guard below did not apply, ~9 GB of 15
    nr_->add_option("--check-features,--check_features", na_.checkFeatures,
                    "--check: N features alive at once, 1..4 (time and VRAM of each; stops at a failure or at 80 % of the DXGI budget)")
        ->check(CLI::Range(1, 4));
    nr_->add_option("--json", na_.json, "write the run summary / diagnostics to this JSON file");

    patch_ = app.add_subcommand("nr-patch", "patch your nvngx_dlssnr.dll for RTX 20/30/40 with dlssnr-patcher (external tool, CUDA 13.3) into bin/nvidia/");
    patch_->add_option("-i,--input", pa_.input, "the original nvngx_dlssnr.dll")->required()->check(CLI::ExistingFile);
    patch_->add_option("-o,--output", pa_.output, "output DLL (default: bin/nvidia/nvngx_dlssnr.dll next to the executable)");
    patch_->add_option("--patcher", pa_.patcher, "dlssnr_patcher.py or its folder (default: DLSSNR_PATCHER_ROOT, tools/dlssnr-patcher)");
    patch_->add_option("--cuda-bin,--cuda_bin", pa_.cudaBin, "folder with ptxas / fatbinary / cuobjdump of CUDA Toolkit 13.3 (default: CUDA_PATH_V13_3/bin)");
    patch_->add_option("--python", pa_.python, "python interpreter (default: DLSSVID_PYTHON or python)");
    patch_->add_option("--arch", pa_.arch, "ada | ampere | turing | blackwell | all, comma separated")->default_val("all");
    patch_->add_flag("--dry-run,--dry_run", pa_.dryRun, "compile and validate without writing the output");
}

int NrCommands::Dispatch() {
    if (nr_ && nr_->parsed()) return CmdNr(na_, *nr_);
    if (patch_ && patch_->parsed()) return CmdPatch(pa_);
    return -1;
}

}  // namespace dlssvid::cli
