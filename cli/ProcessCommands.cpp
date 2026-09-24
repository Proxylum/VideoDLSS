#include "ProcessCommands.h"

#include <windows.h>

#include <cstdio>
#include <fstream>
#include <sstream>

#include "gpu/D3D12Device.h"
#include "pipeline/PassFingerprint.h"
#include "pipeline/ProcessRunner.h"
#include "stages/nr/INrBackend.h"
#include "util/Error.h"
#include "util/Log.h"
#include "viewport/Project.h"

namespace dlssvid::cli {

namespace {

void Progress(const std::string& stage, int64_t done, int64_t total) {
    if (done % 10 == 0 || done == total) {
        if (total > 0) std::fprintf(stderr, "\r%s: %lld/%lld frames", stage.c_str(), static_cast<long long>(done), static_cast<long long>(total));
        else std::fprintf(stderr, "\r%s: %lld frames", stage.c_str(), static_cast<long long>(done));
        if (done == total) std::fprintf(stderr, "\n");
    }
}

std::vector<std::string> Split(const std::string& s, char sep) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string part;
    while (std::getline(ss, part, sep))
        if (!part.empty()) out.push_back(part);
    return out;
}

// Common: stages from the project / defaults, --stages filter, aliases and --param.
void BuildStages(const std::string& projectFile, const std::string& stageList, const std::vector<std::string>& params, const std::string& depthBackend,
                 const std::string& flowBackend, const std::string& upscaleBackend, const std::string& nrBackend, const std::string& fgBackend, double scale,
                 int multiplier, ProcessOptions& o, std::filesystem::path* projectInput, std::filesystem::path* projectPasses, std::filesystem::path* projectResult,
                 int* projectKeep = nullptr, std::string* projectCodec = nullptr, std::map<std::string, std::string>* projectCodecOptions = nullptr) {
    if (!projectFile.empty()) {
        Project p = Project::Load(projectFile);
        o.stages = StagesFromProject(p);
        o.sourceHash = p.SourceHash();  // the project's cache: no rehash when the file is unchanged
        if (projectInput) *projectInput = p.sourceVideo;
        if (projectPasses) *projectPasses = p.passesRoot;
        if (projectResult) *projectResult = p.resultVideo;
        if (projectKeep) *projectKeep = p.passVersionsKeep;
        if (projectCodec) *projectCodec = p.codec;
        if (projectCodecOptions) *projectCodecOptions = p.codecOptions;
    } else {
        o.stages = DefaultProcessStages();
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
                s.params[key] = value;
                return;
            }
        o.stages.push_back(ProcessStage{stage, true, {{key, value}}});
    };
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
        std::printf("%-10s %-8s %-9s %-9s  %s%s%s\n", s.name.c_str(), s.action.c_str(), ShortFingerprint(s.fingerprint).c_str(), onDisk.c_str(), s.kind.c_str(),
                    s.detail.empty() ? "" : ": ", s.detail.c_str());
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
}

int CmdProcess(const ProcessCommands::ProcessArgs& a) {
    ProcessOptions o;
    std::filesystem::path projectInput, projectPasses, projectResult;
    int projectKeep = -1;
    std::string projectCodec;
    std::map<std::string, std::string> projectCodecOptions;
    BuildStages(a.project, a.stages, a.params, a.depthBackend, a.flowBackend, a.upscaleBackend, a.nrBackend, a.fgBackend, a.scale, a.multiplier, o, &projectInput,
                &projectPasses, &projectResult, &projectKeep, &projectCodec, &projectCodecOptions);
    o.keepVersions = a.keepVersions >= 0 ? a.keepVersions : (projectKeep >= 0 ? projectKeep : 2);
    for (const auto& name : Split(a.force, ',')) o.forceStages.insert(name);
    o.input = !a.input.empty() ? std::filesystem::path(a.input) : projectInput;
    if (o.input.empty()) Throw("process: -i/--input (or --project with a source video) is required");
    o.output = !a.output.empty() ? std::filesystem::path(a.output) : projectResult;
    if (o.output.empty()) o.output = o.input.parent_path() / (o.input.stem().string() + "_result.mp4");
    o.passesRoot = !a.passes.empty() ? std::filesystem::path(a.passes) : projectPasses;
    o.skipExisting = !a.noSkipExisting;
    o.disableUnavailable = a.disableUnavailable;
    o.codec = !a.codecGiven && !projectCodec.empty() ? projectCodec : a.codec;  // the project's encoder unless --codec says otherwise
    if (a.codecOptions.empty()) o.codecOptions = projectCodecOptions;
    for (const auto& kv : a.codecOptions) {
        const auto eq = kv.find('=');
        if (eq == std::string::npos) Throw("--codec-opt expects key=value, got: " + kv);
        o.codecOptions[kv.substr(0, eq)] = kv.substr(eq + 1);
    }
    o.hwaccel = a.hwaccel;
    o.frames = a.frames;
    o.warp = a.warp;
    o.passthrough = a.passthrough;
    o.gpuRoundtrip = !a.noGpuRoundTrip;
    o.progress = &Progress;
    if (a.plan) {
        const ProcessPlan plan = PlanProcess(o);
        PrintPlan(plan);
        if (!a.json.empty()) std::ofstream(a.json) << plan.ToJson().dump(2) << "\n";
        return 0;
    }
    const ProcessResult r = RunProcess(o);
    PrintReport(r);
    if (!a.json.empty()) std::ofstream(a.json) << r.ToJson().dump(2) << "\n";
    return 0;
}

int CmdBench(const ProcessCommands::BenchArgs& a) {
    ProcessOptions o;
    BuildStages("", a.stages, a.params, "", "", "", "", "", 0.0, 0, o, nullptr, nullptr, nullptr);
    o.input = a.input;
    wchar_t tmp[MAX_PATH];
    const std::filesystem::path scratch = !a.passes.empty() ? std::filesystem::path(a.passes)
                                                            : std::filesystem::path(GetTempPathW(MAX_PATH, tmp) ? tmp : L".") / ("dlssvid-bench-" + std::to_string(GetCurrentProcessId()));
    o.passesRoot = scratch / "passes";
    o.output = !a.output.empty() ? std::filesystem::path(a.output) : scratch / "result.mkv";  // MKV takes any codec (ffv1, nvenc)
    o.skipExisting = false;
    o.disableUnavailable = true;
    o.codec = a.codec;
    o.frames = a.frames;
    o.warp = a.warp;
    o.progress = &Progress;
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
        std::ofstream(a.json) << j.dump(2) << "\n";
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
    process_->add_option("--stages", pa_.stages, "comma list of stages to run (depth,flow,upscale,nr,fg); default: the project's enabled stages");
    process_->add_option("--param", pa_.params, "stage parameter stage.key=value (repeatable), e.g. nr.intensity=1.2, fg.backend=rife");
    process_->add_option("--depth-backend,--depth_backend", pa_.depthBackend, "da3 | vda | worker | stub");
    process_->add_option("--flow-backend,--flow_backend", pa_.flowBackend, "ofa | searaft | stub");
    process_->add_option("--upscale-backend,--upscale_backend", pa_.upscaleBackend, "dlss | nis | bicubic | trt");
    process_->add_option("--nr-backend,--nr_backend", pa_.nrBackend, "ngx | stub");
    process_->add_option("--fg-backend,--fg_backend", pa_.fgBackend, "dlssg | rife | blend");
    process_->add_option("--scale", pa_.scale, "upscale factor 1.5 | 2 | 3");
    process_->add_option("--multiplier", pa_.multiplier, "frame generation multiplier 2 | 3 | 4");
    process_->add_flag("--no-skip-existing,--no_skip_existing", pa_.noSkipExisting, "recompute every stage even when its pass is complete");
    process_->add_flag("--disable-unavailable,--disable_unavailable", pa_.disableUnavailable, "nr / fg without DLL or GPU: skip the stage instead of failing");
    process_->add_option("--codec", pa_.codec, "video encoder: h264_nvenc | hevc_nvenc | av1_nvenc | ffv1")->default_val("h264_nvenc");
    process_->add_option("--codec-opt", pa_.codecOptions, "encoder private option key=value (repeatable)");
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

    bench_ = app.add_subcommand("bench", "run the pipeline on N frames into a scratch folder and print ms/frame per stage (ТЗ §9)");
    bench_->add_option("-i,--input", ba_.input, "input video file")->required()->check(CLI::ExistingFile);
    bench_->add_option("--frames", ba_.frames, "frames to process")->default_val(30);
    bench_->add_option("--stages", ba_.stages, "comma list of stages (default: all)");
    bench_->add_option("--param", ba_.params, "stage parameter stage.key=value (repeatable)");
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
    if (bench_ && bench_->parsed()) return CmdBench(ba_);
    return -1;
}

}  // namespace dlssvid::cli
