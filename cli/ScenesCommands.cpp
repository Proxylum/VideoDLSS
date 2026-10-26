#include "ScenesCommands.h"

#include <cstdio>
#include <filesystem>
#include <fstream>

#include "io/VideoDecoder.h"
#include "passes/ScenesPass.h"
#include "pipeline/PassVersions.h"
#include "pipeline/ScenesPlan.h"
#include "stages/ParamSchema.h"
#include "stages/scenes/ScenesStage.h"
#include "util/Error.h"
#include "util/Log.h"
#include "util/Sha256.h"

namespace dlssvid::cli {
namespace {

std::string Timecode(int64_t frame, Rational fps) {
    const double t = fps.num > 0 ? static_cast<double>(frame) * fps.den / fps.num : 0.0;
    const int m = static_cast<int>(t / 60.0);
    char buf[32];
    std::snprintf(buf, sizeof buf, "%02d:%06.3f", m, t - 60.0 * m);
    return buf;
}

int CmdScenes(const ScenesCommands::Args& a) {
    if (a.input.empty()) Throw("scenes: -i/--input is required");
    if (a.output.empty()) Throw("scenes: -o/--output is required (the passes root: the pass goes to <output>/scenes)");
    if (const auto problems = ValidateStageParams("scenes", {{"threshold", a.threshold}, {"backend", a.backend}}); !problems.empty()) {
        std::string all;
        for (const auto& p : problems) all += (all.empty() ? "" : "; ") + p;
        Throw("scenes: " + all);
    }
    VideoDecoder decoder(a.input);
    const auto& info = decoder.Info();
    ScenesStageOptions o;
    o.threshold = a.threshold;
    o.backend = a.backend;
    o.outputDir = a.output;
    o.sourceFile = std::filesystem::path(a.input).filename().string();
    o.sourceHash = "sha256:" + Sha256File(a.input);
    o.sourceFormat = info.decoded.Name();
    const int64_t needed = a.frames > 0 ? a.frames : 0;
    std::optional<ScenesSource> scores;
    if (!a.reanalyze) scores = FindSceneScores(a.output, o.sourceHash, o.sourceFormat, needed);
    // the pass on disk goes to the history, as process does: a version a later run would restore is not overwritten
    if (const auto id = RetirePassVersion(a.output, kScenesPass)) Log()->info("scenes: the previous pass kept as version {}", *id);
    ScenesRunResult r;
    if (scores) {
        Log()->info("scenes: scores reused from {}", scores->version.empty() ? std::string("the current pass") : "version " + scores->version);
        r = RederiveScenes(scores->manifest, scores->scores, o);
    } else if (a.threshold <= 0.0) {
        r = WriteOffScenes(o, info.width, info.height, info.frameRate, info.frameCount);
    } else {
        const int64_t total = a.frames > 0 ? a.frames : info.frameCount;
        r = AnalyseScenes(decoder, o, a.frames, [total](int64_t n) {
            if (n % 100 == 0 || n == total) std::fprintf(stderr, "\rscenes: %lld/%lld frames", static_cast<long long>(n), static_cast<long long>(total));
        });
        std::fprintf(stderr, "\n");
    }
    std::printf("scenes: %lld frames, threshold %.2f (%s, %.1f ms/frame)\n", static_cast<long long>(r.list.frames), a.threshold,
                r.rederived ? "stored scores, no decoding" : r.list.off ? "off" : "decoded", r.msPerFrame);
    const size_t shown = std::min<size_t>(r.list.cuts.size(), 20);
    for (size_t i = 0; i < shown; ++i)
        std::printf("  cut  %6lld  %s\n", static_cast<long long>(r.list.cuts[i]), Timecode(r.list.cuts[i], info.frameRate).c_str());
    if (r.list.cuts.size() > shown) std::printf("  ... %zu more\n", r.list.cuts.size() - shown);
    for (const auto& [f, s] : r.list.nearMisses) std::printf("  near %6lld  %s  score %.2f\n", static_cast<long long>(f), Timecode(f, info.frameRate).c_str(), s);
    std::printf("scenes: %zu cut(s) -> %s\n", r.list.cuts.size(), r.outDir.string().c_str());
    if (!a.json.empty()) {
        nlohmann::json nearMisses = nlohmann::json::array();
        for (const auto& [f, s] : r.list.nearMisses) nearMisses.push_back({f, s});
        const nlohmann::json j = {{"cuts", r.list.cuts}, {"count", r.list.cuts.size()}, {"id", r.list.id},           {"threshold", a.threshold},
                                  {"frames", r.list.frames}, {"reached_end", r.list.reachedEnd}, {"rederived", r.rederived}, {"off", r.list.off},
                                  {"near", nearMisses},          {"ms_per_frame", r.msPerFrame},  {"pass", r.outDir.string()}};
        std::ofstream(a.json) << j.dump(2) << "\n";
    }
    return 0;
}

}  // namespace

void ScenesCommands::Register(CLI::App& app) {
    scenes_ = app.add_subcommand("scenes", "the clip's scene-cut list -> <output>/scenes (a data pass: the cuts and every frame's score; read by flow, upscale, nr, fg)");
    scenes_->add_option("-i,--input", a_.input, "input video file")->check(CLI::ExistingFile);
    scenes_->add_option("-o,--output", a_.output, "pass root folder (writes scenes/)");
    scenes_->add_option("--threshold", a_.threshold, "scene-cut threshold 0..1: mean |dY| of neighbouring frames above it starts a new shot (0 = off)")
        ->default_val(0.24);
    scenes_->add_option("--backend", a_.backend, "luma")->default_val("luma");
    scenes_->add_option("--frames", a_.frames, "analyse at most N frames from the start")->default_val(-1);
    scenes_->add_flag("--reanalyze", a_.reanalyze, "decode the source again even when the stored scores stand for it");
    scenes_->add_option("--json", a_.json, "write the list and the run to this JSON file");
}

int ScenesCommands::Dispatch() {
    if (scenes_ && scenes_->parsed()) return CmdScenes(a_);
    return -1;
}

}  // namespace dlssvid::cli
