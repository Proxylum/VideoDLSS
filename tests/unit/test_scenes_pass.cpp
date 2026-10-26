#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "passes/ScenesPass.h"
#include "pipeline/PassFingerprint.h"
#include "pipeline/ProcessRunner.h"

using namespace dlssvid;

namespace {
std::filesystem::path Root(const char* name) {
    const auto root = std::filesystem::path(DLSSVID_TEST_TMP) / "scenes_pass" / name;
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root);
    return root;
}
Manifest Base() {
    Manifest m;
    m.width = 1920;
    m.height = 800;
    m.fps = {24, 1};
    m.sourceFile = "clip.mp4";
    m.sourceHash = "sha256:src";
    m.stageParams = {{"backend", "luma"}};
    return m;
}
SceneList List(std::vector<int64_t> cuts, int64_t frames, double threshold = 0.24) {
    SceneList l;
    l.cuts = std::move(cuts);
    l.id = SceneCutsId(l.cuts);
    l.frames = frames;
    l.threshold = threshold;
    l.reachedEnd = true;
    return l;
}
// rewrites stage_params of the pass in dir (a broken or hand-made pass)
void Patch(const std::filesystem::path& dir, const char* key, const nlohmann::json& value) {
    Manifest m = Manifest::Load(dir);
    m.stageParams[key] = value;
    m.Save(dir);
}
}  // namespace

// PR-13: the clip's cut list - its identity is its content ("cuts:<sha256>"), none for no cut: the stages below read it as an input
// only when it is not empty (spec §2d: the fingerprint of nr does not count it while it is empty).
TEST_CASE("Scene list: id of the content, cuts from the scores, near misses", "[unit][scenes]") {
    CHECK(SceneCutsId({}).empty());
    const std::string a = SceneCutsId({3, 40});
    CHECK(a.rfind("cuts:", 0) == 0);
    CHECK(a.size() == 5 + 64);
    CHECK(SceneCutsId({3, 40}) == a);  // deterministic
    CHECK(SceneCutsId({3, 41}) != a);
    CHECK(SceneCutsId({3}) != a);
    CHECK(ShortFingerprint(a).size() == 8);

    const std::vector<float> scores = {0.9f, 0.01f, 0.24f, 0.5f, 0.2f, 0.15f, 0.3f, 0.1f};
    CHECK(CutsFromScores(scores, 0.24) == std::vector<int64_t>{3, 6});  // frame 0 never
    // a score at the threshold is no cut (strict >) and a near miss (<=): 0.5 is exact in float32, 0.24 is not (review of PR-13)
    CHECK(CutsFromScores(scores, 0.5) == std::vector<int64_t>{});
    REQUIRE(!NearCuts(scores, 0.5).empty());
    CHECK(NearCuts(scores, 0.5)[0] == std::pair<int64_t, float>{3, 0.5f});
    CHECK(CutsFromScores(scores, 0.0).empty());                          // 0: off
    CHECK(CutsFromScores(scores, 0.6).empty());
    CHECK(CutsFromScores({}, 0.24).empty());
    const auto nearMisses = NearCuts(scores, 0.24);  // [0.144, 0.24]: 2 (0.24), 4 (0.2), 5 (0.15), highest first
    REQUIRE(nearMisses.size() == 3);
    CHECK(nearMisses[0].first == 2);
    CHECK(nearMisses[1].first == 4);
    CHECK(nearMisses[2].first == 5);
    CHECK(NearCuts(scores, 0.24, 1).size() == 1);
    CHECK(NearCuts(scores, 0.0).empty());
}

TEST_CASE("Scenes pass: a data pass written and read back; a broken or foreign list is no list", "[unit][scenes][regression]") {
    const auto dir = Root("rw") / "scenes";
    SceneList l = List({3, 40}, 100);
    l.nearMisses = {{12, 0.21f}};
    SceneScores s;
    s.scores.assign(100, 0.01f);
    s.scores[3] = 0.4f;
    s.scores[40] = 0.5f;
    s.analysis = {{"algorithm", kScenesAlgorithm}, {"max_width", 640}, {"ms_per_frame", 6.1}};
    WriteScenesPass(dir, Base(), l, &s);

    const Manifest m = Manifest::Load(dir);
    CHECK(m.IsDataPass());
    CHECK(m.pass == "scenes");
    CHECK(m.width == 1920);
    CHECK(m.frameCount == 100);
    CHECK(m.stageParams["backend"] == "luma");  // base's stage_params kept
    CHECK(m.stageParams["cuts_id"] == l.id);
    CHECK(m.stageParams["analysis"]["scores_count"] == 100);
    CHECK(std::filesystem::exists(dir / kScenesScoresFile));

    std::string why;
    const auto r = ReadSceneList(dir, "sha256:src", &why);
    INFO(why);
    REQUIRE(r);
    CHECK(r->cuts == std::vector<int64_t>{3, 40});
    CHECK(r->id == l.id);
    CHECK(r->threshold == 0.24);
    CHECK(r->frames == 100);
    CHECK(r->reachedEnd);
    CHECK(r->IsCut(40));
    CHECK_FALSE(r->IsCut(4));
    CHECK(r->CountIn(0, 39) == 1);
    REQUIRE(r->nearMisses.size() == 1);
    CHECK(r->nearMisses[0].first == 12);
    CHECK(ReadSceneList(dir, "")) ;  // no hash given: not checked
    CHECK(ReadSceneListExpect(dir, l.id, "sha256:src").cuts.size() == 2);
    CHECK_THROWS(ReadSceneListExpect(dir, SceneCutsId({3}), "sha256:src"));  // the list changed under a run

    const auto sc = ReadSceneScores(dir, m, &why);
    INFO(why);
    REQUIRE(sc);
    CHECK(sc->scores.size() == 100);
    CHECK(sc->scores[40] == 0.5f);
    CHECK(sc->analysis["algorithm"] == kScenesAlgorithm);
    CHECK(sc->analysis["ms_per_frame"] == 6.1);

    // what makes no list
    CHECK_FALSE(ReadSceneList(dir, "sha256:other", &why));
    CHECK(why == "of another source");
    CHECK_FALSE(ReadSceneList(Root("none") / "scenes", "", &why));
    CHECK(why == "absent");
    struct Bad {
        const char* key;
        nlohmann::json value;
        const char* why;
    };
    for (const Bad& b : {Bad{"scene_cuts", {40, 3}, "unsorted or repeated"}, Bad{"scene_cuts", {3, 3}, "unsorted or repeated"},
                         Bad{"scene_cuts", {0, 3}, "outside frames"}, Bad{"scene_cuts", {3, 100}, "outside frames"}, Bad{"cuts_id", "cuts:00", "cuts_id"},
                         Bad{"ab", {{"mv_scale", "auto"}}, "A/B"}, Bad{"scene_cuts", "3", "not a list"},
                         // a field of another type: a broken list, never an exception out of the planner (review of PR-13)
                         Bad{"threshold", "0.24", "another type"}, Bad{"off", 0, "another type"}, Bad{"cuts_id", nullptr, "another type"},
                         Bad{"analysis", 5, "another type"}}) {
        INFO(b.key << " = " << b.value.dump());
        WriteScenesPass(dir, Base(), l, &s);
        Patch(dir, b.key, b.value);
        // the list's own id with it: only the check of the order or the range can refuse it (review of PR-13)
        if (std::string(b.key) == "scene_cuts" && b.value.is_array()) Patch(dir, "cuts_id", SceneCutsId(b.value.get<std::vector<int64_t>>()));
        CHECK_FALSE(ReadSceneList(dir, "sha256:src", &why));
        CHECK(why.find(b.why) != std::string::npos);
    }
    WriteScenesPass(dir, Base(), l, &s);
    {  // a part of the clip
        Manifest p = Manifest::Load(dir);
        p.firstFrame = 10;
        p.Save(dir);
        CHECK_FALSE(ReadSceneList(dir, "sha256:src"));
    }
    // without scores: the list stands, the scores do not
    WriteScenesPass(dir, Base(), l, nullptr);
    CHECK(ReadSceneList(dir, "sha256:src"));
    CHECK_FALSE(std::filesystem::exists(dir / kScenesScoresFile));  // the old file is not left beside another list
    CHECK_FALSE(ReadSceneScores(dir, Manifest::Load(dir)));
    // scores changed on disk: not used
    WriteScenesPass(dir, Base(), l, &s);
    std::ofstream(dir / kScenesScoresFile, std::ios::binary | std::ios::app) << "x";
    CHECK_FALSE(ReadSceneScores(dir, Manifest::Load(dir), &why));
    CHECK(why.find("sha256") != std::string::npos);
    // an empty list: no id, a list all the same
    WriteScenesPass(dir, Base(), List({}, 100), &s);
    const auto empty = ReadSceneList(dir, "sha256:src");
    REQUIRE(empty);
    CHECK(empty->cuts.empty());
    CHECK(empty->id.empty());
}

TEST_CASE("Scenes pass: complete for the frames it covers, a label of the list a consumer used", "[unit][scenes]") {
    const auto dir = Root("complete") / "scenes";
    WriteScenesPass(dir, Base(), List({3}, 100), nullptr);
    CHECK(PassComplete(dir, 100, "sha256:src"));
    CHECK(PassComplete(dir, 60, "sha256:src"));
    CHECK_FALSE(PassComplete(dir, 101, "sha256:src"));
    CHECK_FALSE(PassComplete(dir, 100, "sha256:other"));
    CHECK(PassComplete(dir, -1, "sha256:src"));  // the length unknown: the analysis reached the end
    SceneList part = List({3}, 50);
    part.reachedEnd = false;
    WriteScenesPass(dir, Base(), part, nullptr);
    CHECK_FALSE(PassComplete(dir, -1, "sha256:src"));
    CHECK(PassComplete(dir, 50, "sha256:src"));
    SceneList off = List({}, 0, 0.0);
    off.off = true;
    off.reachedEnd = false;
    WriteScenesPass(dir, Base(), off, nullptr);
    CHECK(PassComplete(dir, 100, "sha256:src"));  // threshold 0: no length needed

    Manifest nr;
    nr.pass = "color_nr";
    nr.stageParams = {{"scene_cuts_id", "cuts:ab"}};
    CHECK(SceneLabelOf(nr) == "cuts:ab");
    nr.stageParams = {{"scene_cut_count", 2}};  // PR-6: reset by its own detector
    CHECK(SceneLabelOf(nr) == "pr6");
    nr.stageParams = {{"scene_cut_count", 0}};
    CHECK(SceneLabelOf(nr).empty());
    Manifest fg;
    fg.pass = "color_fg";
    fg.stageParams = {{"scene_cut_count", 2}};
    CHECK(SceneLabelOf(fg).empty());
}
