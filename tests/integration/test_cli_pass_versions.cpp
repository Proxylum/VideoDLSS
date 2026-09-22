// `dlssvid process --plan` and `dlssvid passes list|use|gc` as processes (stage 9, MR A).

#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "TestClips.h"
#include "passes/Manifest.h"
#include "util/Subprocess.h"
#include "viewport/Project.h"

using namespace dlssvid;
using namespace dlssvid::test;

namespace {

struct CliResult {
    uint32_t code = 0;
    std::string out;
};

CliResult RunCli(std::vector<std::string> args) {
    args.insert(args.begin(), DLSSVID_CLI_PATH);
    CliResult r;
    r.code = Subprocess::Run(args, &r.out);
    return r;
}

std::filesystem::path Dir(const char* name) {
    const auto d = TempDir() / "cli_versions" / name;
    std::filesystem::remove_all(d);
    std::filesystem::create_directories(d);
    return d;
}

nlohmann::json ReadJson(const std::filesystem::path& p) {
    std::ifstream in(p);
    nlohmann::json j;
    in >> j;
    return j;
}

std::vector<std::string> operator+(std::vector<std::string> a, const std::vector<std::string>& b) {
    a.insert(a.end(), b.begin(), b.end());
    return a;
}

}  // namespace

TEST_CASE("dlssvid process --plan decides without running; passes list / use / gc manage the versions", "[integration][cli][process][versions]") {
    ClipSpec spec;
    spec.frames = 4;
    spec.width = 48;
    spec.height = 32;
    const auto dir = Dir("flow");
    const auto clip = WriteClip(dir / "clip.mkv", spec);
    const auto root = dir / "passes";
    const std::vector<std::string> common{"process", "-i", clip.string(), "-o", (dir / "result.mkv").string(), "--passes", root.string(), "--warp", "--codec", "ffv1",
                                          "--param", "depth.backend=stub", "--param", "flow.backend=stub", "--param", "upscale.backend=nis", "--param", "nr.backend=stub",
                                          "--param", "fg.backend=blend"};

    const CliResult plan = RunCli(common + std::vector<std::string>{"--plan", "--json", (dir / "plan.json").string()});
    CHECK(plan.code == 0);
    CHECK(plan.out.find("plan: 5 stage(s) to run, 0 reused") != std::string::npos);
    CHECK(plan.out.find("missing") != std::string::npos);
    const nlohmann::json pj = ReadJson(dir / "plan.json");
    REQUIRE(pj["stages"].size() == 5);
    CHECK(pj["stages"][0]["action"] == "run");
    CHECK(pj["stages"][0]["kind"] == "missing");
    CHECK(pj["run_count"] == 5);
    CHECK(pj["final_pass"] == "color_fg");
    CHECK(!std::filesystem::exists(root));
    CHECK(!std::filesystem::exists(dir / "result.mkv"));

    const CliResult run = RunCli(common + std::vector<std::string>{"--json", (dir / "run.json").string()});
    CHECK(run.code == 0);
    const nlohmann::json rj = ReadJson(dir / "run.json");
    CHECK(rj["stages"][3]["status"] == "ran");
    CHECK(rj["stages"][3]["fingerprint"] == pj["stages"][3]["fingerprint"]);
    CHECK(rj["stages"][3]["retired"] == "");
    CHECK(run.out.find("[") != std::string::npos);  // the short fingerprint in the report

    const CliResult plan2 = RunCli(common + std::vector<std::string>{"--plan"});
    CHECK(plan2.code == 0);
    CHECK(plan2.out.find("plan: 0 stage(s) to run, 5 reused") != std::string::npos);

    const CliResult list = RunCli({"passes", "list", "--passes", root.string(), "--json", (dir / "list.json").string()});
    CHECK(list.code == 0);
    CHECK(list.out.find("color_nr") != std::string::npos);
    CHECK(list.out.find("current") != std::string::npos);
    CHECK(list.out.find("total:") != std::string::npos);
    const nlohmann::json lj = ReadJson(dir / "list.json");
    REQUIRE(lj["versions"].size() == 7);
    for (const auto& v : lj["versions"]) {
        CHECK(v["current"] == true);
        CHECK(v["version"] == "current");
        CHECK(!v["fingerprint"].get<std::string>().empty());
        CHECK(v["frames"].get<int>() >= 4);  // color_fg: 7 = FgFrameCount(4, 2)
        CHECK(v["bytes"].get<uint64_t>() > 0);
    }

    // a new intensity: nr and fg get new versions, the old ones are kept
    const CliResult run2 = RunCli(common + std::vector<std::string>{"--param", "nr.intensity=1.5", "--json", (dir / "run2.json").string()});
    CHECK(run2.code == 0);
    const nlohmann::json rj2 = ReadJson(dir / "run2.json");
    CHECK(rj2["stages"][0]["status"] == "reused");
    CHECK(rj2["stages"][3]["status"] == "ran");
    CHECK(rj2["stages"][3]["decision"]["kind"] == "params_changed");
    const std::string oldId = rj2["stages"][3]["retired"];
    CHECK(!oldId.empty());
    CHECK(run2.out.find("params_changed") != std::string::npos);

    // --force: one stage recomputed in place (same fingerprint), the others reused; bad parameters are refused before the run
    const CliResult forced = RunCli(common + std::vector<std::string>{"--param", "nr.intensity=1.5", "--force", "fg", "--json", (dir / "force.json").string()});
    CHECK(forced.code == 0);
    const nlohmann::json fj = ReadJson(dir / "force.json");
    CHECK(fj["stages"][4]["status"] == "ran");
    CHECK(fj["stages"][4]["decision"]["kind"] == "forced");
    CHECK(fj["stages"][4]["retired"] == "");
    CHECK(fj["stages"][3]["status"] == "reused");
    CHECK(RunCli(common + std::vector<std::string>{"--force", "bogus"}).code != 0);
    CHECK(RunCli(common + std::vector<std::string>{"--param", "nr.intensity=9", "--plan"}).code != 0);

    const CliResult list2 = RunCli({"passes", "list", "--passes", root.string(), "--pass", "color_nr", "--json", (dir / "list2.json").string()});
    CHECK(list2.code == 0);
    const nlohmann::json lj2 = ReadJson(dir / "list2.json");
    REQUIRE(lj2["versions"].size() == 2);
    CHECK(lj2["versions"][1]["version"] == oldId);
    const std::string oldFp = lj2["versions"][1]["fingerprint"], newFp = lj2["versions"][0]["fingerprint"];
    CHECK(rj["stages"][3]["fingerprint"] == oldFp);
    CHECK(lj2["versions"][1]["params"]["intensity"] == 1);
    CHECK(lj2["versions"][0]["params"]["intensity"] == 1.5);

    // switch color_nr back by a prefix of the id: the 1.5 version goes to the history
    const CliResult use = RunCli({"passes", "use", "color_nr", oldId.substr(0, 12), "--passes", root.string(), "--json", (dir / "use.json").string()});
    CHECK(use.code == 0);
    CHECK(use.out.find("is now current") != std::string::npos);
    CHECK(Manifest::Load(root / "color_nr").fingerprint == oldFp);
    const nlohmann::json uj = ReadJson(dir / "use.json");
    REQUIRE(uj["switched"].size() == 1);
    const std::string newId = uj["switched"][0]["retired"];
    CHECK(!newId.empty());
    // the same by stage name (nr has one pass), with the full id of the 1.5 version
    const CliResult useStage = RunCli({"passes", "use", "nr", newId, "--passes", root.string()});
    CHECK(useStage.code == 0);
    CHECK(Manifest::Load(root / "color_nr").fingerprint == newFp);
    // and back through `process`: the plan restores the 1.0 version instead of recomputing
    const CliResult plan3 = RunCli(common + std::vector<std::string>{"--plan"});
    CHECK(plan3.code == 0);
    CHECK(plan3.out.find("restore") != std::string::npos);
    CHECK(plan3.out.find("plan: 0 stage(s) to run") != std::string::npos);

    // gc: dry run first, then for real
    const CliResult dry = RunCli({"passes", "gc", "--passes", root.string(), "--keep", "1", "--dry-run", "--json", (dir / "gc.json").string()});
    CHECK(dry.code == 0);
    CHECK(dry.out.find("would remove") != std::string::npos);
    const nlohmann::json gj = ReadJson(dir / "gc.json");
    CHECK(gj["removed"].size() == 2);  // color_nr and color_fg histories
    CHECK(gj["dry_run"].get<bool>());
    CHECK(std::filesystem::exists(root / "color_nr.v"));
    const CliResult gc = RunCli({"passes", "gc", "--passes", root.string(), "--keep", "1"});
    CHECK(gc.code == 0);
    CHECK(gc.out.find("removed color_nr") != std::string::npos);
    CHECK(!std::filesystem::exists(root / "color_nr.v"));
    CHECK(!std::filesystem::exists(root / "color_fg.v"));
    CHECK(Manifest::Exists(root / "color_nr"));

    // the project supplies the root and pass_versions_keep
    Project p = Project::Create(clip, root);
    p.passVersionsKeep = 1;
    p.codec = "ffv1";  // the project's encoder is used when --codec is absent
    p.resultVideo = dir / "project_result.mkv";
    for (auto& st : p.stages) {
        if (st.name == "depth" || st.name == "flow" || st.name == "nr") st.params["backend"] = "stub";
        if (st.name == "upscale") st.params["backend"] = "nis";
        if (st.name == "fg") st.params["backend"] = "blend";
    }
    p.Save(dir / "clip.dlssvid.json");
    CHECK(RunCli({"process", "--project", (dir / "clip.dlssvid.json").string(), "--warp"}).code == 0);
    CHECK(std::filesystem::exists(dir / "project_result.mkv"));
    const CliResult viaProject = RunCli({"passes", "list", "--project", (dir / "clip.dlssvid.json").string()});
    CHECK(viaProject.code == 0);
    CHECK(viaProject.out.find("color_fg") != std::string::npos);
    const CliResult gcProject = RunCli({"passes", "gc", "--project", (dir / "clip.dlssvid.json").string()});
    CHECK(gcProject.code == 0);
    CHECK(gcProject.out.find("1 kept per pass") != std::string::npos);

    // errors
    CHECK(RunCli({"passes", "use", "color_nr", "nope", "--passes", root.string()}).code != 0);
    CHECK(RunCli({"passes", "use", "depth", "nope", "--passes", root.string()}).code != 0);
    CHECK(RunCli({"passes", "list"}).code != 0);
    CHECK(RunCli({"passes", "gc", "--project", (dir / "missing.json").string()}).code != 0);
    CHECK(RunCli({"passes"}).code != 0);
}
