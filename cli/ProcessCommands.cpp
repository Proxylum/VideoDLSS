#include "ProcessCommands.h"

#include <io.h>
#include <windows.h>

#include <cstdio>
#include <fstream>
#include <sstream>

#include "gpu/D3D12Device.h"
#include "io/OutputFormat.h"
#include "pipeline/PassFingerprint.h"
#include "pipeline/ProcessRunner.h"
#include "stages/ParamSchema.h"
#include "stages/StagePreset.h"
#include "stages/nr/INrBackend.h"
#include "util/Error.h"
#include "util/Log.h"
#include "util/Paths.h"
#include "viewport/PipelinePreset.h"
#include "viewport/Project.h"

namespace dlssvid::cli {

namespace {

void Progress(const std::string& stage, int64_t done, int64_t total) {
    if (done % 10 == 0 || done == total) {
        // A terminal gets one updating line; a pipe (the GUI's TaskQueue, tests) gets whole lines, so that a progress
        // update never glues onto a log line such as "process: fg failed - ..." that the GUI parses.
        static const bool tty = _isatty(_fileno(stderr)) != 0;
        const char* lead = tty ? "\r" : "";
        if (total > 0) std::fprintf(stderr, "%s%s: %lld/%lld frames", lead, stage.c_str(), static_cast<long long>(done), static_cast<long long>(total));
        else std::fprintf(stderr, "%s%s: %lld frames", lead, stage.c_str(), static_cast<long long>(done));
        if (!tty || done == total) std::fprintf(stderr, "\n");
    }
}

// Reports carry paths and log text that may not be valid UTF-8 (ACP paths, a child's log): never fail on them.
std::string DumpJson(const nlohmann::json& j) { return j.dump(2, ' ', false, nlohmann::json::error_handler_t::replace); }

std::vector<std::string> Split(const std::string& s, char sep) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string part;
    while (std::getline(ss, part, sep))
        if (!part.empty()) out.push_back(part);
    return out;
}

// What the command line takes from the project file and a pipeline preset (layers 1 and 2, docs/conventions.md «--param-file»).
struct FileLayer {
    std::filesystem::path input, passes, result;  // the project's (empty without one)
    int keep = -1;
    bool encoder = false;  // a project or a pipeline preset gave the encoder
    std::string codec;
    std::map<std::string, std::string> codecOptions;
    OutputFormatRequest format;
    bool settingsNote = true;
};

bool IsProcessStage(const std::string& name) {
    return std::find(ProcessStageOrder().begin(), ProcessStageOrder().end(), name) != ProcessStageOrder().end();
}

std::string Quoted(const std::vector<std::string>& keys) {
    std::string s;
    for (const auto& k : keys) s += (s.empty() ? "'" : ", '") + k + "'";
    return s;
}

// Common: the stages and the encoder in layers (PR-9): the project or the built-in defaults < a pipeline preset (--param-file without
// stage=) < --stages < the stage files (--param-file stage=..., a card preset without stage=), in command-line order < the backend
// options, --scale, --multiplier < --param. A stage file switches its stage on, as --param does.
void BuildStages(const std::string& projectFile, const std::vector<std::string>& paramFiles, const std::string& stageList, const std::vector<std::string>& params,
                 const std::string& depthBackend, const std::string& flowBackend, const std::string& upscaleBackend, const std::string& nrBackend,
                 const std::string& fgBackend, double scale, int multiplier, ProcessOptions& o, FileLayer* layer) {
    Project base;
    if (!projectFile.empty()) {
        base = Project::Load(projectFile);
        o.sourceHash = base.SourceHash();  // the project's cache: no rehash when the file is unchanged
        if (o.sourceHash.empty()) o.sourceHash = base.sourceHash;  // the file is gone (moved): what the project last saw
    } else {
        base.stages = Project::DefaultStages();
    }
    // the --param-file list: one pipeline preset (layer 2, wherever it stands) and the stage files (layer 4, in order)
    struct StageFile {
        std::string stage, file, spec;
    };
    std::vector<StageFile> stageFiles;
    std::string pipelineFile;
    bool presetEncoder = false;  // the pipeline preset has an encode section (without one the project's or the built-in encoder stays)
    for (const std::string& spec : paramFiles) {
        const size_t eq = spec.find('=');
        const std::string head = eq == std::string::npos ? std::string() : spec.substr(0, eq);
        if (IsProcessStage(head)) {  // the stage form only when the text before the first '=' is a stage: a path with '=' stays a path
            if (eq + 1 == spec.size()) Throw("--param-file " + spec + ": no file after " + head + "=");
            stageFiles.push_back({head, spec.substr(eq + 1), spec});
            continue;
        }
        const PresetLoad l = LoadPreset(spec);  // without stage=: a preset only
        if (l.Has(PresetIssueKind::NotPreset))
            Throw("--param-file " + spec + ": a JSON object of stage settings needs stage=<file.json> (e.g. nr=" + spec +
                  "); without stage= the file must be a dlssvid-preset");
        if (!l.ok()) Throw("--param-file " + spec + ": " + l.problemsText());
        if (l.preset.scope == PresetScope::Stage) {  // a card preset: to its stage, as <stage>=<file>
            Log()->info("--param-file {}: a card preset of {} ('{}'), applied to {}", spec, l.preset.stage(), l.preset.name, l.preset.stage());
            stageFiles.push_back({l.preset.stage(), spec, spec});
            continue;
        }
        if (!pipelineFile.empty()) Throw("--param-file: one pipeline preset at a time (" + pipelineFile + ", " + spec + ")");
        pipelineFile = spec;
        presetEncoder = l.preset.encode.has_value();
        std::vector<PresetIssue> notes = l.issues;
        ApplyPipelinePreset(base, l.preset, &notes);  // the stages it names (switch, settings) and the encoder
        for (const auto& n : notes) Log()->warn("--param-file {}: pipeline preset '{}': {}", spec, l.preset.name, n.text);
    }
    o.stages = StagesFromProject(base);
    if (layer) {
        if (!projectFile.empty()) {
            layer->input = base.sourceVideo;
            layer->passes = base.passesRoot;
            layer->result = base.resultVideo;  // with the container of a pipeline preset's codec
            layer->keep = base.passVersionsKeep;
        }
        layer->encoder = !projectFile.empty() || presetEncoder;
        if (layer->encoder) {
            layer->codec = base.codec;
            layer->codecOptions = base.codecOptions;
            layer->format = base.outputFormat;
            layer->settingsNote = base.settingsNote;
        }
    }
    if (!stageList.empty()) {
        const std::vector<std::string> only = Split(stageList, ',');
        for (auto& s : o.stages) s.enabled = std::find(only.begin(), only.end(), s.name) != only.end();
        for (const auto& name : only) {
            if (std::find(ProcessStageOrder().begin(), ProcessStageOrder().end(), name) == ProcessStageOrder().end()) Throw("--stages: unknown stage '" + name + "'");
            bool found = false;
            for (const auto& s : o.stages) found = found || s.name == name;
            if (!found) o.stages.push_back(ProcessStage{name, true, nlohmann::json::object()});
        }
    }
    auto alias = [&](const std::string& stage, const std::string& key, const nlohmann::json& value) {
        for (auto& s : o.stages)
            if (s.name == stage) {
                // another backend: a model from the old backend's list goes (a project with VDA Large + --depth-backend da3)
                if (key == "backend" && value.is_string()) s.params = ParamsForBackendChange(stage, s.params, value.get<std::string>());
                s.params[key] = value;
                return;
            }
        o.stages.push_back(ProcessStage{stage, true, {{key, value}}});
    };
    for (const StageFile& sf : stageFiles) {
        StageParamsFile f = ReadStageParamsFile(sf.stage, sf.file);  // throws: unreadable, > 1 MiB, broken, a moved key, a refused preset
        if (!f.dropped.empty())
            Throw("--param-file " + sf.spec + ": " + Quoted(f.dropped) + " not " + sf.stage + " settings (stages." + sf.stage + ".params keys)");
        if (f.preset && f.scope == PresetScope::Pipeline)
            Log()->warn("--param-file {}: the pipeline preset '{}' gives its {} settings only; its other stages, switches and encode are not applied (give it "
                        "without {}= for the whole pipeline)",
                        sf.spec, f.presetName, sf.stage, sf.stage);
        for (const auto& n : f.notes)  // a pipeline preset's other stages and fields: said by the line above; a card preset's: here
            if (f.scope != PresetScope::Pipeline || (n.kind != PresetIssueKind::OtherStages && n.kind != PresetIssueKind::PipelineFields))
                Log()->warn("--param-file {}: preset '{}': {}", sf.spec, f.presetName, n.text);
        ProcessStage* s = nullptr;
        for (auto& x : o.stages)
            if (x.name == sf.stage) s = &x;
        if (!s) {
            o.stages.push_back(ProcessStage{sf.stage, true, nlohmann::json::object()});
            s = &o.stages.back();
        }
        if (f.preset) {  // its keys replace the form's (and, from a pipeline preset, the tuning keys); the others stay
            std::vector<std::string> ignored;
            s->params = ApplyPreset(sf.stage, s->params, f.params, f.scope, &ignored);
            const std::string backend = StageBackend(sf.stage, s->params);
            for (const auto& k : ignored) {
                const ParamSpec* spec = FindStageSchema(sf.stage) ? FindStageSchema(sf.stage)->Find(k) : nullptr;
                if (spec && IsListlessKey(*spec, backend))
                    Log()->warn("--param-file {}: {}.{} has no list for the backend {}: presets do not set it, the project's value stays (give --param)", sf.spec,
                                sf.stage, k, backend);
                else
                    Log()->warn("--param-file {}: {}.{} does not apply to the backend {}, skipped", sf.spec, sf.stage, k, backend);
            }
        } else {  // a settings object merges over the stage's; its backend first, as the aliases
            if (f.params.contains("backend") && f.params["backend"].is_string())
                s->params = ParamsForBackendChange(sf.stage, s->params, f.params["backend"].get<std::string>());
            for (const auto& [k, v] : f.params.items()) s->params[k] = v;
        }
        s->enabled = true;  // as --param
    }
    if (!depthBackend.empty()) alias("depth", "backend", depthBackend);
    if (!flowBackend.empty()) alias("flow", "backend", flowBackend);
    if (!upscaleBackend.empty()) alias("upscale", "backend", upscaleBackend);
    if (!nrBackend.empty()) alias("nr", "backend", nrBackend);
    if (!fgBackend.empty()) alias("fg", "backend", fgBackend);
    if (scale > 0) alias("upscale", "scale", scale);
    if (multiplier > 0) alias("fg", "multiplier", multiplier);
    for (const auto& spec : params) ApplyParamSpec(o.stages, spec);
}

void PrintPlan(const ProcessPlan& p) {
    std::printf("%-10s %-8s %-9s %-9s  %s\n", "stage", "action", "needs", "on disk", "why");
    for (const auto& s : p.stages) {
        const std::string onDisk = !s.current ? "-" : s.currentFingerprint.empty() ? "legacy" : ShortFingerprint(s.currentFingerprint);
        // PR-13: a provisional decision says so - the list is analysed first, the run decides on it
        const char* mark = s.name == "scenes" && s.action == "run" && !s.sceneCuts.known ? " (pending: the scene list is analysed first)"
                           : s.pending && s.pendingRule == "no_cut"                    ? " (pending: reused if no cut is found)"
                           : s.pending && s.pendingRule == "same_list"                 ? " (pending: reused if the same cuts are found)"
                           : s.pending                                                 ? " (pending: reused if the stages above are)"
                                                                                       : "";
        std::printf("%-10s %-8s %-9s %-9s  %s%s%s%s\n", s.name.c_str(), s.action.c_str(), ShortFingerprint(s.fingerprint).c_str(), onDisk.c_str(), s.kind.c_str(),
                    s.detail.empty() ? "" : ": ", s.detail.c_str(), mark);
    }
    for (const auto& s : p.stages) {  // the advice of the stages (TASK-0039: licence, VRAM)
        for (const auto& w : s.warnings) std::printf("warning: %s: %s\n", s.name.c_str(), w.c_str());
        if (!s.info.empty()) std::printf("note: %s: %s\n", s.name.c_str(), s.info.c_str());
        if (!s.error.empty()) std::printf("error: %s: %s\n", s.name.c_str(), s.error.c_str());
    }
    // PR-13: the scene-cut list and the stages that reset at it
    const StageDecision* scenes = nullptr;
    std::string readers, unlisted;
    for (const auto& s : p.stages) {
        if (s.name == "scenes") scenes = &s;
        else if (s.inputs.count("scenes")) readers += (readers.empty() ? "" : ", ") + s.name;
        else if (StageReadsSceneCuts(s.name, s.params)) unlisted += (unlisted.empty() ? "" : ", ") + s.name;
    }
    if (scenes && scenes->action == "run" && !scenes->sceneCuts.known) {
        std::printf("note: scenes: not analysed yet; the stages below are reused if the list comes out as the one they were made with\n");
    } else if (scenes && scenes->sceneCuts.known && !readers.empty()) {
        std::string cuts;
        for (size_t i = 0; i < scenes->sceneCuts.cuts.size() && i < 10; ++i) cuts += (cuts.empty() ? "" : ", ") + std::to_string(scenes->sceneCuts.cuts[i]);
        if (scenes->sceneCuts.cuts.size() > 10) cuts += ", ...";
        std::printf("note: scenes: %lld cut(s) (%s); reset by %s\n", static_cast<long long>(scenes->sceneCuts.count), cuts.c_str(), readers.c_str());
    } else if (!scenes && !readers.empty()) {
        std::printf("note: scenes: the list on disk is used; reset by %s\n", readers.c_str());
    } else if (!scenes && !unlisted.empty()) {
        std::printf("note: scenes is off and no scene list is on disk: %s do not reset at cuts\n", unlisted.c_str());
    }
    std::printf("plan: %d stage(s) to run, %d reused; result from %s; passes in %s\n", p.runCount, p.reuseCount,
                p.finalPass.empty() ? "the source (passthrough)" : p.finalPass.c_str(), p.passesRoot.string().c_str());
}

void PrintReport(const ProcessResult& r) {
    std::printf("%-12s %-12s %8s %10s %12s  %s\n", "stage", "status", "frames", "seconds", "ms/frame", "output");
    for (const auto& s : r.stages)
        std::printf("%-12s %-12s %8lld %10.1f %12.1f  %s%s%s%s\n", s.name.c_str(), s.status.c_str(), static_cast<long long>(s.frames), s.seconds, s.msPerFrame,
                    s.outDir.string().c_str(), s.fingerprint.empty() ? "" : (" [" + ShortFingerprint(s.fingerprint) + "]").c_str(), s.reason.empty() ? "" : " — ",
                    s.reason.c_str());
    std::printf("result:      %s (%lld frames at %.3f fps%s) in %.1f s\n", r.output.string().c_str(), static_cast<long long>(r.framesOut), r.outputFps.ToDouble(),
                r.audio ? ", audio copied" : "", r.seconds);
    std::printf("format:      %s %s (%s)\n", r.codec.c_str(), r.outputFormat.Describe().c_str(), r.outputPixelFormat.c_str());
    for (const auto& w : r.warnings) std::printf("warning:     %s\n", w.c_str());
}

int CmdProcess(const ProcessCommands::ProcessArgs& a) {
    ProcessOptions o;
    FileLayer file;
    BuildStages(a.project, a.paramFiles, a.stages, a.params, a.depthBackend, a.flowBackend, a.upscaleBackend, a.nrBackend, a.fgBackend, a.scale, a.multiplier, o,
                &file);
    std::filesystem::path projectInput = file.input, projectPasses = file.passes, projectResult = file.result;
    o.settingsNote = file.settingsNote && !a.noSettingsNote;
    o.keepVersions = a.keepVersions >= 0 ? a.keepVersions : (file.keep >= 0 ? file.keep : 2);
    for (const auto& name : Split(a.force, ',')) o.forceStages.insert(name);
    o.input = !a.input.empty() ? std::filesystem::path(a.input) : projectInput;
    if (o.input.empty()) Throw("process: -i/--input (or --project with a source video) is required");
    // --project P -i other.mp4: P's stages and encoder for another video. The hash, passes and result of P belong to P's
    // video: before, the other file got P's cached hash, so P's passes were reused for it and P's result overwritten
    // (TASK-0043). Another video is hashed itself and gets its own passes and result unless --passes / -o say otherwise;
    // P's video moved elsewhere (the same content) stays P's.
    if (!a.input.empty() && !projectInput.empty()) {
        std::error_code ec;
        const bool samePath = std::filesystem::equivalent(o.input, projectInput, ec) && !ec;
        if (!samePath) {
            const std::string inputHash = "sha256:" + Sha256FileCached(o.input);
            if (inputHash != o.sourceHash) {  // also when the project knows no hash of its video: it cannot be told apart
                Log()->info("process: {} is not the source of the project: its own hash, passes and result", o.input.string());
                projectPasses.clear();
                projectResult.clear();
            }
            o.sourceHash = inputHash;
        }
    }
    // the project's (or a pipeline preset's) encoder unless --codec says otherwise
    o.codec = !a.codecGiven && file.encoder && !file.codec.empty() ? file.codec : a.codec;
    // the project's result takes the container of the codec --codec gives it (PR-9): a ProRes preset, then --codec h264_nvenc,
    // writes .mp4, not H.264 into the .mov the preset renamed it to; and ProRes of --codec no longer goes into the project's .mp4. Without
    // --codec the path stays the project's (a pipeline preset renamed it already)
    const bool codecChanged = a.codecGiven && o.codec != (file.encoder ? file.codec : std::string());
    o.output = !a.output.empty() ? std::filesystem::path(a.output) : codecChanged ? ResultPathFor(projectResult, o.codec) : projectResult;
    if (o.output.empty()) o.output = o.input.parent_path() / (o.input.stem().string() + "_result" + ResultExtensionFor(o.codec));
    o.passesRoot = !a.passes.empty() ? std::filesystem::path(a.passes) : projectPasses;
    o.skipExisting = !a.noSkipExisting;
    o.disableUnavailable = a.disableUnavailable;
    if (a.codecOptions.empty()) o.codecOptions = file.codecOptions;  // --codec-opt replaces them all, a preset's bitrate too
    for (const auto& kv : a.codecOptions) {
        const auto eq = kv.find('=');
        if (eq == std::string::npos) Throw("--codec-opt expects key=value, got: " + kv);
        o.codecOptions[kv.substr(0, eq)] = kv.substr(eq + 1);
    }
    o.outputFormat = file.format;  // the project's (a pipeline preset's) bit depth / chroma unless the flags say otherwise
    if (!a.bitDepth.empty()) {
        const auto depth = ParseBitDepthChoice(a.bitDepth);
        if (!depth) Throw("--bit-depth must be auto | 8 | 10 | 12 (got '" + a.bitDepth + "')");
        o.outputFormat.bitDepth = *depth;
    }
    if (!a.chroma.empty()) {
        if (!ValidChromaChoice(a.chroma)) Throw("--chroma must be auto | 420 | 422 | 444 (got '" + a.chroma + "')");
        o.outputFormat.chroma = a.chroma;
    }
    o.hwaccel = a.hwaccel;
    o.frames = a.frames;
    o.warp = a.warp;
    o.passthrough = a.passthrough;
    o.gpuRoundtrip = !a.noGpuRoundTrip;
    o.progress = &Progress;
    o.fgExecutable = a.fgInProcess ? std::filesystem::path{} : CurrentExecutable();  // audit fg-2: FG in its own process
    o.fgCrashAfter = a.fgCrashAfter;
    o.fgCrashOnExit = a.fgCrashOnExit;
    o.fgStall = a.fgStall;
    if (a.plan) {
        {  // the VRAM advice needs the GPU's memory, as RunProcess has it
            D3D12Device device({o.warp, false});
            o.vramBytes = device.IsWarp() ? 0 : device.DedicatedVideoMemory();
        }
        const ProcessPlan plan = PlanProcess(o);
        PrintPlan(plan);
        // PR-9: what the CLI decided about the encoder (layer 2 cannot be checked otherwise without a run)
        std::string options;
        for (const auto& [k, v] : o.codecOptions) options += (options.empty() ? "" : " ") + k + "=" + v;
        std::printf("encode: %s, bit depth %s, chroma %s, %s, settings note %s, output %s\n", o.codec.c_str(), BitDepthChoiceName(o.outputFormat.bitDepth).c_str(),
                    o.outputFormat.chroma.c_str(), options.empty() ? "no options" : options.c_str(), o.settingsNote ? "on" : "off", PathText(o.output).c_str());
        if (!a.json.empty()) {
            nlohmann::json j = plan.ToJson();
            j["encode"] = {{"codec", o.codec},
                           {"options", o.codecOptions},
                           {"bit_depth", BitDepthChoiceName(o.outputFormat.bitDepth)},
                           {"chroma", o.outputFormat.chroma},
                           {"settings_note", o.settingsNote},
                           {"output", PathText(o.output)}};
            std::ofstream(a.json) << DumpJson(j) << "\n";
        }
        for (const auto& s : plan.stages)
            if (!s.error.empty()) return 1;  // the run would be refused
        return 0;
    }
    const ProcessResult r = RunProcess(o);
    PrintReport(r);
    if (!a.json.empty()) std::ofstream(a.json) << DumpJson(r.ToJson()) << "\n";
    return 0;
}

int CmdBench(const ProcessCommands::BenchArgs& a) {
    ProcessOptions o;
    FileLayer file;
    BuildStages("", a.paramFiles, a.stages, a.params, "", "", "", "", "", 0.0, 0, o, &file);
    o.input = a.input;
    wchar_t tmp[MAX_PATH];
    const std::filesystem::path scratch = !a.passes.empty() ? std::filesystem::path(a.passes)
                                                            : std::filesystem::path(GetTempPathW(MAX_PATH, tmp) ? tmp : L".") / ("dlssvid-bench-" + std::to_string(GetCurrentProcessId()));
    o.passesRoot = scratch / "passes";
    o.output = !a.output.empty() ? std::filesystem::path(a.output) : scratch / "result.mkv";  // MKV takes any codec (ffv1, nvenc)
    o.skipExisting = false;
    o.disableUnavailable = true;
    o.codec = !a.codecGiven && file.encoder ? file.codec : a.codec;  // a pipeline preset's encoder unless --codec says otherwise
    if (file.encoder) {
        o.codecOptions = file.codecOptions;
        o.settingsNote = file.settingsNote;
        // bench has no --bit-depth / --chroma: over another codec the preset's layout could be one it cannot encode, found only at the
        // encode after every stage ran - auto then
        if (!a.codecGiven || a.codec == file.codec) o.outputFormat = file.format;
        else Log()->info("bench: --codec {} over the preset's {}: bit depth and chroma auto", a.codec, file.codec);
    }
    o.frames = a.frames;
    o.warp = a.warp;
    o.progress = &Progress;
    o.fgExecutable = CurrentExecutable();  // as `process`: the fg timing includes its process, start-up reported apart
    std::string gpu, arch, driver;
    {
        D3D12Device device({a.warp, false});
        gpu = device.AdapterName();
        arch = GpuArchitectureFromName(gpu);
        driver = NvidiaDriverFromUmd(device.UmdDriverVersion()).ToString();
    }
    const ProcessResult r = RunProcess(o);
    std::printf("GPU:          %s (%s), driver %s\n", gpu.c_str(), arch.empty() ? "?" : arch.c_str(), driver.c_str());
    std::printf("input:        %s, %lld frames\n", a.input.c_str(), static_cast<long long>(r.framesIn));
    PrintReport(r);
    if (!a.json.empty()) {
        nlohmann::json j = r.ToJson();
        j["gpu"] = gpu;
        j["architecture"] = arch;
        j["driver"] = driver;
        std::ofstream(a.json) << DumpJson(j) << "\n";
    }
    if (!a.keep && a.passes.empty()) {
        std::error_code ec;
        std::filesystem::remove_all(scratch, ec);
    }
    return 0;
}

}  // namespace

void ProcessCommands::Register(CLI::App& app) {
    process_ = app.add_subcommand("process", "run the whole pipeline (depth -> flow -> upscale -> nr -> fg over the pass cache) and encode the result with audio");
    process_->add_option("-i,--input", pa_.input, "input video file (default: the project's source)")->check(CLI::ExistingFile);
    process_->add_option("-o,--output", pa_.output, "output video file (default: the project's result or <input>_result.mp4)");
    process_->add_option("--passes", pa_.passes, "pass cache root (default: the project's or <output>_passes)");
    process_->add_option("--project", pa_.project, "*.dlssvid.json: stages and their parameters as the GUI saved them")->check(CLI::ExistingFile);
    std::string names;
    for (const auto& n : PipelineStageNames()) names += (names.empty() ? "" : ",") + n;
    process_->add_option("--stages", pa_.stages, "comma list of stages to run (" + names + "); default: the project's enabled stages");
    process_->add_option("--param", pa_.params, "stage parameter stage.key=value (repeatable), e.g. nr.intensity=1.2, fg.backend=rife");
    const char* paramFileHelp =
        "settings from a file (repeatable). stage=<file.json>: a JSON object of the stage's keys (stages.<stage>.params) merged over the project's, or a "
        "dlssvid-preset whose settings of that stage replace its form keys (a pipeline preset's also its tuning keys: backend, pass format ...; the "
        "keys of the machine stay); switches the stage on like --param. <file.json> without "
        "stage=: a dlssvid-preset - a pipeline preset sets the stages it names (on/off and settings) and the encoder as a project would, a card preset goes "
        "to its stage. --stages, the backend options, --codec / --bit-depth / --chroma / --codec-opt and --param win";
    process_->add_option("--param-file,--param_file", pa_.paramFiles, paramFileHelp);
    process_->add_option("--depth-backend,--depth_backend", pa_.depthBackend, "da3 | vda | worker | worker:vda | stub");
    process_->add_option("--flow-backend,--flow_backend", pa_.flowBackend, "ofa | searaft | stub");
    process_->add_option("--upscale-backend,--upscale_backend", pa_.upscaleBackend, "dlss | nis | bicubic | trt | worker");
    process_->add_option("--nr-backend,--nr_backend", pa_.nrBackend, "ngx | stub");
    process_->add_option("--fg-backend,--fg_backend", pa_.fgBackend, "dlssg | rife | blend");
    process_->add_option("--scale", pa_.scale, "upscale factor 1.5 | 2 | 3");
    process_->add_option("--multiplier", pa_.multiplier, "frame generation multiplier 2 | 3 | 4");
    process_->add_flag("--no-skip-existing,--no_skip_existing", pa_.noSkipExisting, "recompute every stage even when its pass is complete");
    process_->add_flag("--no-settings-note,--no_settings_note", pa_.noSettingsNote,
                       "do not write the settings note (what made the result: the comment tag, in .mkv also DLSSVID_SETTINGS) into the result");
    process_->add_flag("--disable-unavailable,--disable_unavailable", pa_.disableUnavailable,
                       "nr / fg without DLL or GPU, or an fg process that fails: skip the stage (fg: encode color_nr) instead of failing");
    process_->add_option("--codec", pa_.codec, "video encoder: h264_nvenc | hevc_nvenc | av1_nvenc | prores_ks | ffv1 (default: the project's, else h264_nvenc)")
        ->default_val("h264_nvenc");
    process_->add_option("--codec-opt", pa_.codecOptions, "encoder private option key=value (repeatable)");
    process_->add_option("--bit-depth,--bit_depth", pa_.bitDepth,
                         "result bit depth: auto | 8 | 10 | 12 (auto: the source's within the codec — 10-bit HEVC / AV1 / ProRes / FFV1, 8-bit H.264)");
    process_->add_option("--chroma", pa_.chroma, "result chroma subsampling: auto | 420 | 422 | 444 (auto: 4:2:0 for NVENC, 4:2:2 for ProRes, the source's for FFV1)");
    process_->add_option("--hwaccel", pa_.hwaccel, "source decode for the encode / passthrough step: none | cuda")->default_val("none");
    process_->add_option("--frames", pa_.frames, "process at most N source frames")->default_val(-1);
    process_->add_flag("--warp", pa_.warp, "use the WARP software adapter");
    process_->add_flag("--passthrough", pa_.passthrough, "decode -> GPU -> encode without processing (stage 0)");
    process_->add_flag("--no-gpu-roundtrip", pa_.noGpuRoundTrip, "passthrough: skip the GPU upload/readback");
    process_->add_flag("--plan", pa_.plan, "decide which stages would run or be reused (fingerprints, versions) and exit without processing");
    process_->add_option("--force", pa_.force, "comma list of stages to recompute even when their pass matches (nr,fg)");
    process_->add_option("--keep-versions,--keep_versions", pa_.keepVersions,
                         "pass versions kept per pass after the run, the current one included; 0 = keep all (default: the project's pass_versions_keep or 2)")
        ->default_val(-1);
    process_->add_option("--json", pa_.json, "write the stage report (or the plan with --plan) to this JSON file");
    // hidden: debugging and the isolation tests of the fg process (audit fg-2)
    process_->add_flag("--fg-in-process,--fg_in_process", pa_.fgInProcess, "debugging: run FG in this process, without isolation")->group("");
    process_->add_option("--fg-crash-after,--fg_crash_after", pa_.fgCrashAfter, "isolation test: the fg process terminates itself after N real frames")
        ->default_val(-1)
        ->group("");
    process_->add_flag("--fg-crash-on-exit,--fg_crash_on_exit", pa_.fgCrashOnExit, "isolation test: the fg process terminates itself after its result")->group("");
    process_->add_flag("--fg-stall,--fg_stall", pa_.fgStall, "isolation test: the fg process hangs before reading the input")->group("");

    bench_ = app.add_subcommand("bench", "run the pipeline on N frames into a scratch folder and print ms/frame per stage (ТЗ §9)");
    bench_->add_option("-i,--input", ba_.input, "input video file")->required()->check(CLI::ExistingFile);
    bench_->add_option("--frames", ba_.frames, "frames to process")->default_val(30);
    bench_->add_option("--stages", ba_.stages, "comma list of stages (default: all)");
    bench_->add_option("--param", ba_.params, "stage parameter stage.key=value (repeatable)");
    bench_->add_option("--param-file,--param_file", ba_.paramFiles,
                       "settings from a file (repeatable). stage=<file.json>: a JSON object of the stage's keys merged over the built-in defaults, or a "
                       "dlssvid-preset whose settings of that stage replace its form keys (a pipeline preset's also its tuning keys). <file.json> without "
                       "stage=: a dlssvid-preset - a pipeline preset sets the stages it names and the encoder, a card preset goes to its stage. --stages, "
                       "--codec and --param win; --codec over a pipeline preset's codec takes bit depth and chroma auto");
    bench_->add_option("--passes", ba_.passes, "keep the passes here instead of a temporary folder");
    bench_->add_option("-o,--output", ba_.output, "result video (default: temporary)");
    bench_->add_option("--codec", ba_.codec, "video encoder")->default_val("h264_nvenc");
    bench_->add_flag("--warp", ba_.warp, "use the WARP software adapter");
    bench_->add_flag("--keep", ba_.keep, "keep the temporary folder");
    bench_->add_option("--json", ba_.json, "write the report to this JSON file");
}

int ProcessCommands::Dispatch() {
    if (process_ && process_->parsed()) {
        pa_.codecGiven = process_->count("--codec") > 0;
        return CmdProcess(pa_);
    }
    if (bench_ && bench_->parsed()) {
        ba_.codecGiven = bench_->count("--codec") > 0;
        return CmdBench(ba_);
    }
    return -1;
}

}  // namespace dlssvid::cli
