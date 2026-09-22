// Pass versions (stage 9, MR A) on synthetic pass folders: ids, retiring the current version to <pass>.v/, listing,
// switching versions (prefixes, errors, partial folders), gc that keeps the newest N and the referenced ones.

#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>

#include "passes/Manifest.h"
#include "pipeline/PassVersions.h"

using namespace dlssvid;

namespace {

std::filesystem::path Root(const char* name) {
    const std::filesystem::path d = std::filesystem::path(DLSSVID_TEST_TMP) / "pass_versions" / name;
    std::filesystem::remove_all(d);
    std::filesystem::create_directories(d);
    return d;
}

// A pass folder with a manifest and tiny dummy frame files (only their presence and size matter here).
void MakePass(const std::filesystem::path& root, const std::string& pass, const std::string& fp, const std::string& created, int frames,
              const std::map<std::string, std::string>& inputs = {}, const nlohmann::json& params = nlohmann::json::object()) {
    const auto kind = ParsePassKind(pass);
    Manifest m = Manifest::ForPass(kind ? *kind : PassKind::ColorSr, 4, 4, FileFormat::Raw);
    m.pass = pass;
    m.frameCount = frames;
    m.firstFrame = 0;
    m.lastFrame = frames - 1;
    m.sourceHash = "sha256:src";
    m.fingerprint = fp;
    m.created = created;
    m.inputs = inputs;
    m.paramsCanonical = params;
    m.tool = {{"backend", "stub"}};
    m.Save(root / pass);
    for (int i = 0; i < frames; ++i) std::ofstream(root / pass / m.FrameFileName(i), std::ios::binary) << std::string(16, 'x');
}

std::vector<std::string> Ids(const std::vector<PassVersionInfo>& v) {
    std::vector<std::string> out;
    for (const auto& x : v) out.push_back(x.current ? "current" : x.id);
    return out;
}

}  // namespace

TEST_CASE("PassVersionId, RetirePassVersion and ListPassVersions", "[unit][versions]") {
    const auto root = Root("retire");
    MakePass(root, "color_nr", "sha256:aaaa1111ffff", "2026-09-22T14:02:00Z", 3, {}, {{"intensity", 1}});
    CHECK(PassVersionId(Manifest::Load(root / "color_nr"), root / "color_nr") == "20260922-140200_aaaa1111");
    CHECK(PassHistoryDir(root, "color_nr") == root / "color_nr.v");
    CHECK(PassesUnderRoot(root) == std::vector<std::string>{"color_nr"});
    const auto cur = CurrentPassVersion(root, "color_nr");
    REQUIRE(cur);
    CHECK(cur->current);
    CHECK(cur->frames == 3);
    CHECK(cur->bytes > 0);
    CHECK(cur->params == nlohmann::json({{"intensity", 1}}));
    CHECK(cur->ToJson()["version"] == "current");
    CHECK(!CurrentPassVersion(root, "color_fg"));

    const auto id = RetirePassVersion(root, "color_nr");
    REQUIRE(id);
    CHECK(*id == "20260922-140200_aaaa1111");
    CHECK(!std::filesystem::exists(root / "color_nr"));
    CHECK(Manifest::Exists(root / "color_nr.v" / *id));
    CHECK(!RetirePassVersion(root, "color_nr"));  // nothing current any more
    CHECK(PassesUnderRoot(root).empty());
    CHECK(Ids(ListPassVersions(root)) == std::vector<std::string>{*id});  // a history without a current version is still listed

    MakePass(root, "color_nr", "sha256:bbbb2222ffff", "2026-09-22T14:09:00Z", 3, {}, {{"intensity", 1.4}});
    const auto all = ListPassVersions(root, "color_nr");
    REQUIRE(all.size() == 2);
    CHECK(all[0].current);
    CHECK(all[0].fingerprint == "sha256:bbbb2222ffff");
    CHECK(all[1].id == *id);
    CHECK(all[1].ToJson()["version"] == *id);
    CHECK(ListPassVersions(root).size() == 2);

    // the same time and fingerprint again: the id stays unique
    REQUIRE(RetirePassVersion(root, "color_nr") == "20260922-140900_bbbb2222");
    MakePass(root, "color_nr", "sha256:aaaa1111ffff", "2026-09-22T14:02:00Z", 3);
    CHECK(RetirePassVersion(root, "color_nr") == "20260922-140200_aaaa1111-2");
    // a legacy pass (no fingerprint, no created) gets its id from the manifest file time
    MakePass(root, "color_nr", "", "", 3);
    const auto legacy = RetirePassVersion(root, "color_nr");
    REQUIRE(legacy);
    CHECK(legacy->size() == 15 + 1 + 6);
    CHECK(legacy->substr(15) == "_legacy");
    // a folder without a manifest (an interrupted run) is moved aside as partial
    std::filesystem::create_directories(root / "color_nr");
    std::ofstream(root / "color_nr" / "stray.bin") << "x";
    const auto partial = RetirePassVersion(root, "color_nr");
    REQUIRE(partial);
    CHECK(partial->substr(15) == "_partial");
    CHECK(std::filesystem::exists(root / "color_nr.v" / *partial / "stray.bin"));
}

TEST_CASE("UsePassVersion swaps folders, resolves unique prefixes and rejects unknown or ambiguous ids", "[unit][versions]") {
    const auto root = Root("use");
    MakePass(root, "color_nr", "sha256:aaaa1111ffff", "2026-09-22T14:02:00Z", 3, {}, {{"intensity", 1}});
    RetirePassVersion(root, "color_nr");
    MakePass(root, "color_nr", "sha256:bbbb2222ffff", "2026-09-22T14:09:00Z", 3, {}, {{"intensity", 1.4}});

    const auto retired = UsePassVersion(root, "color_nr", "20260922-1402");  // prefix
    REQUIRE(retired);
    CHECK(*retired == "20260922-140900_bbbb2222");
    CHECK(CurrentPassVersion(root, "color_nr")->fingerprint == "sha256:aaaa1111ffff");
    CHECK(Ids(ListPassVersions(root, "color_nr")) == std::vector<std::string>{"current", "20260922-140900_bbbb2222"});
    CHECK(std::filesystem::exists(root / "color_nr" / Manifest::Load(root / "color_nr").FrameFileName(2)));

    CHECK_THROWS(UsePassVersion(root, "color_nr", "nope"));
    CHECK_THROWS(UsePassVersion(root, "color_fg", "20260922"));
    // two versions share a prefix
    RetirePassVersion(root, "color_nr");
    CHECK_THROWS(UsePassVersion(root, "color_nr", "20260922-140"));
    CHECK(!std::filesystem::exists(root / "color_nr"));
    // no current version: the restored one just moves in
    CHECK(!UsePassVersion(root, "color_nr", "20260922-140900_bbbb2222"));
    CHECK(CurrentPassVersion(root, "color_nr")->fingerprint == "sha256:bbbb2222ffff");
    // a partial current folder is kept aside as well
    std::filesystem::remove_all(root / "color_nr");
    std::filesystem::create_directories(root / "color_nr");
    std::ofstream(root / "color_nr" / "frame.bin") << "x";
    const auto partial = UsePassVersion(root, "color_nr", "20260922-140200_aaaa1111");
    REQUIRE(partial);
    CHECK(partial->find("_partial") != std::string::npos);
    CHECK(CurrentPassVersion(root, "color_nr")->fingerprint == "sha256:aaaa1111ffff");
}

TEST_CASE("GcPassVersions keeps the newest N per pass, referenced versions and nothing with keep 0", "[unit][versions]") {
    const auto root = Root("gc");
    const std::string n0 = "sha256:00000000n0", n1 = "sha256:11111111n1", n2 = "sha256:22222222n2", n3 = "sha256:33333333n3";
    MakePass(root, "color_nr", n0, "2026-09-22T13:50:00Z", 2);
    RetirePassVersion(root, "color_nr");
    MakePass(root, "color_nr", n1, "2026-09-22T14:00:00Z", 2);
    RetirePassVersion(root, "color_nr");
    MakePass(root, "color_nr", n2, "2026-09-22T14:10:00Z", 2);
    RetirePassVersion(root, "color_nr");
    MakePass(root, "color_nr", n3, "2026-09-22T14:20:00Z", 2);
    MakePass(root, "color_fg", "sha256:ffff0000f2", "2026-09-22T14:21:00Z", 4, {{"color_nr", n1}});
    CHECK(Ids(ListPassVersions(root, "color_nr")) == std::vector<std::string>{"current", "20260922-141000_22222222", "20260922-140000_11111111", "20260922-135000_00000000"});
    const auto refs = ReferencedFingerprints(root);
    CHECK(refs.count(n1) == 1);
    CHECK(refs.count(n0) == 0);

    CHECK(GcPassVersions(root, 0).removed.empty());
    const PassGcReport dry = GcPassVersions(root, 2, "", true);
    REQUIRE(dry.removed.size() == 1);
    CHECK(dry.removed[0].fingerprint == n0);  // n1 stays: color_fg lists it as an input
    CHECK(dry.bytes == dry.removed[0].bytes);
    CHECK(ListPassVersions(root, "color_nr").size() == 4);  // a dry run removes nothing

    const PassGcReport gc2 = GcPassVersions(root, 2);
    REQUIRE(gc2.removed.size() == 1);
    CHECK(gc2.removed[0].fingerprint == n0);
    CHECK(Ids(ListPassVersions(root, "color_nr")) == std::vector<std::string>{"current", "20260922-141000_22222222", "20260922-140000_11111111"});
    CHECK(GcPassVersions(root, 1).removed.size() == 1);  // n2 goes, n1 is referenced
    CHECK(Ids(ListPassVersions(root, "color_nr")) == std::vector<std::string>{"current", "20260922-140000_11111111"});

    // the reference moves on: color_fg's old version still points at n1 while a new current points at n3
    RetirePassVersion(root, "color_fg");
    MakePass(root, "color_fg", "sha256:ffff0000f3", "2026-09-22T14:30:00Z", 4, {{"color_nr", n3}});
    CHECK(GcPassVersions(root, 2).removed.empty());  // both histories have one entry
    const PassGcReport gc1 = GcPassVersions(root, 1);  // color_fg first: its old version goes, which releases n1
    REQUIRE(gc1.removed.size() == 2);
    CHECK(gc1.removed[0].pass == "color_fg");
    CHECK(gc1.removed[1].fingerprint == n1);
    CHECK(Ids(ListPassVersions(root, "color_nr")) == std::vector<std::string>{"current"});
    CHECK(!std::filesystem::exists(root / "color_nr.v"));  // an empty history folder is removed
    CHECK(!std::filesystem::exists(root / "color_fg.v"));

    // a pass with a history but no current version keeps `keep` entries
    MakePass(root, "color_sr", "sha256:s1", "2026-09-22T15:00:00Z", 2);
    RetirePassVersion(root, "color_sr");
    MakePass(root, "color_sr", "sha256:s2", "2026-09-22T15:10:00Z", 2);
    RetirePassVersion(root, "color_sr");
    CHECK(GcPassVersions(root, 1, "color_sr").removed.size() == 1);
    CHECK(Ids(ListPassVersions(root, "color_sr")) == std::vector<std::string>{"20260922-151000_s2"});
}
