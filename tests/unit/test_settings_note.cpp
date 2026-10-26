#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <string>

#include "passes/Manifest.h"
#include "pipeline/PassVersions.h"
#include "pipeline/SettingsNote.h"

using namespace dlssvid;

namespace {
std::filesystem::path Root(const char* name) {
    const auto root = std::filesystem::path(DLSSVID_TEST_TMP) / "settings_note" / name;
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root);
    return root;
}
void Save(const std::filesystem::path& root, const char* pass, PassKind kind, const nlohmann::json& stageParams) {
    Manifest m = Manifest::ForPass(kind, 3840, 2160, FileFormat::Exr);
    m.stageParams = stageParams;
    m.sourceFile = "D:/videos/clip.mp4";  // a path of the machine: never in the note
    m.Save(root / pass);
}
void Chain(const std::filesystem::path& root) {
    Save(root, "color_sr", PassKind::ColorSr, {{"backend", "dlss"}, {"scale", 2.0}, {"quality", "DLAA"}, {"dll_dir", "C:/Program Files/NVIDIA/dlss"}, {"fallback", false}});
    Save(root, "color_nr", PassKind::ColorNr,
         {{"backend", "ngx"}, {"style", 1}, {"intensity", 1.2}, {"passes", 1}, {"local_tone", 1.0}, {"local_structure", 1.0}, {"skin_structure", -1.0},
          {"auto_mask", false}, {"guides", true}, {"scene_cuts_id", "cuts:0123"}, {"scene_cut_threshold", 0.24}, {"scene_cut_count", 3}, {"scene_cuts", {40, 97, 130}},
          {"temporal", 0.0},
          {"model_scale", 1.0}, {"color_source", "color_sr"}, {"diagnostics", {{"dll", "C:/dlss/nvngx_dlssnr.dll"}}},
          {"tonemap", {{"curve", "passthrough"}, {"input_transfer", "srgb"}, {"exposure_ev", 0.0}}}, {"masks", nlohmann::json::array()}});
    Save(root, "color_fg", PassKind::ColorFg, {{"backend", "dlssg"}, {"multiplier", 2}, {"color_source", "color_nr"}});
}
}  // namespace

// PR-11 (spec §2g): the result says what made it - one segment per stage of its chain, the values the stages recorded, an adopted pass
// marked; one ASCII line of at most 1 KB, no paths of the machine.
TEST_CASE("Settings note: one segment per stage that ran, values from stageParams, adopted passes marked, no absolute paths, ASCII, <= 1 KB",
          "[unit][encode]") {
    const auto root = Root("chain");
    Chain(root);
    const std::string note = BuildSettingsNote(root, "color_fg");
    CHECK(note == std::string("VideoDLSS ") + DLSSVID_VERSION +
                      " | scenes: luma threshold=0.24 cuts=3 | upscale: dlss x2 quality=DLAA | nr: ngx style=natural intensity=1.2 passes=1 local_tone=1"
                      " local_structure=1 skin_structure=follow auto_mask=off guides=on temporal=0 model_scale=1 | fg: dlssg x2");
    CHECK(note.find(":/") == std::string::npos);  // no C:/..., D:/...
    CHECK(note.find("nvngx") == std::string::npos);
    CHECK(note.size() <= kSettingsNoteMax);
    for (const char c : note) CHECK((c >= 32 && c <= 126));
    const nlohmann::json j = BuildSettingsJson(root, "color_fg");
    CHECK(j["stages"]["scenes"]["cuts"] == 3);  // PR-13: the list the result was made with, from the first label met (nr here)
    CHECK(j["stages"]["scenes"]["threshold"] == 0.24);
    CHECK_FALSE(j["stages"]["nr"].contains("cuts"));
    CHECK_FALSE(j["stages"]["nr"].contains("scene_cut_threshold"));
    CHECK_FALSE(j["stages"]["upscale"].contains("dll_dir"));
    CHECK_FALSE(j.dump().find(":/") != std::string::npos);

    // the result encoded from color_nr (fg off or failed): no fg, whatever lies on disk
    CHECK(BuildSettingsNote(root, "color_nr").find("fg:") == std::string::npos);
    CHECK(BuildSettingsNote(root, "color_nr").find("upscale: dlss") != std::string::npos);
    // nr read the video: the color_sr on disk is not in the result
    Save(root, "color_nr", PassKind::ColorNr, {{"backend", "stub"}, {"passes", 2}, {"color_source", "video"}, {"adopted", true}});
    const std::string video = BuildSettingsNote(root, "color_fg");
    CHECK(video.find("upscale:") == std::string::npos);
    CHECK(video.find("nr: stub passes=2 (adopted)") != std::string::npos);
    // fg names the pass it read (pass:<folder>, slot:<pass>); before PR-11 "pass" alone: the input process fingerprinted
    Save(root, "color_nr", PassKind::ColorNr, {{"backend", "stub"}, {"color_source", "video"}});
    for (const char* src : {"pass:color_sr", "slot:color_sr"}) {
        Save(root, "color_fg", PassKind::ColorFg, {{"backend", "blend"}, {"multiplier", 3}, {"color_source", src}});
        INFO(src);
        CHECK(BuildSettingsNote(root, "color_fg") == std::string("VideoDLSS ") + DLSSVID_VERSION + " | upscale: dlss x2 quality=DLAA | fg: blend x3");
    }
    {
        Manifest m = Manifest::ForPass(PassKind::ColorFg, 3840, 2160, FileFormat::Exr);
        m.stageParams = {{"backend", "blend"}, {"multiplier", 2}, {"color_source", "pass"}};
        m.inputs["color_nr"] = LegacyPassId(Manifest::Load(root / "color_nr"));  // what process records for the nr pass on disk
        m.fingerprint = "sha256:01";  // inputs are written with the fingerprint (a pass process stamped)
        m.Save(root / "color_fg");
        CHECK(BuildSettingsNote(root, "color_fg").find("| nr: stub | fg: blend x2") != std::string::npos);
    }
    // "pass:" (a folder given with a trailing separator) falls back to the recorded input; another folder is not this root's pass
    {
        Manifest m = Manifest::ForPass(PassKind::ColorFg, 3840, 2160, FileFormat::Exr);
        m.stageParams = {{"backend", "blend"}, {"multiplier", 2}, {"color_source", "pass:"}};
        m.inputs["color_nr"] = LegacyPassId(Manifest::Load(root / "color_nr"));  // what process records for the nr pass on disk
        m.fingerprint = "sha256:01";
        m.Save(root / "color_fg");
        CHECK(BuildSettingsNote(root, "color_fg").find("| nr: stub") != std::string::npos);
        m.stageParams["color_source"] = "pass:nr_elsewhere";
        m.Save(root / "color_fg");
        CHECK(BuildSettingsNote(root, "color_fg").find("nr:") == std::string::npos);
    }
    // the bypass (intensity 0, PR-7): no model - nor its filter or resolution (review of PR-11)
    Save(root, "color_nr", PassKind::ColorNr,
         {{"backend", "ngx"}, {"bypass", true}, {"passes", 0}, {"intensity", 0.0}, {"temporal", 0.3}, {"model_scale", 0.5}, {"color_source", "color_sr"}});
    CHECK(BuildSettingsNote(root, "color_nr").find("nr: ngx (no model) intensity=0") != std::string::npos);
    CHECK(BuildSettingsNote(root, "color_nr").find("temporal=") == std::string::npos);
    CHECK(BuildSettingsNote(root, "color_nr").find("model_scale=") == std::string::npos);
    // what acts only with its switch: the filter's threshold with the filter on, the ratio transfer at another model resolution
    Save(root, "color_nr", PassKind::ColorNr,
         {{"backend", "ngx"}, {"temporal", 0.3}, {"temporal_threshold", 0.2}, {"model_scale", 0.5}, {"transfer", 0.8}, {"max_ratio", 6.0}, {"color_source", "video"}});
    CHECK(BuildSettingsNote(root, "color_nr").find("temporal=0.3 temporal_threshold=0.2 model_scale=0.5 transfer=0.8 max_ratio=6") != std::string::npos);
    Save(root, "color_nr", PassKind::ColorNr,
         {{"backend", "ngx"}, {"temporal", 0.0}, {"temporal_threshold", 0.2}, {"model_scale", 1.0}, {"max_ratio", 6.0}, {"color_source", "video"}});
    CHECK(BuildSettingsNote(root, "color_nr").find("threshold") == std::string::npos);
    CHECK(BuildSettingsNote(root, "color_nr").find("max_ratio") == std::string::npos);
    // nothing encoded from a colour pass (passthrough), or no manifest: the app alone
    CHECK(BuildSettingsNote(root, "") == std::string("VideoDLSS ") + DLSSVID_VERSION);
    CHECK(BuildSettingsNote(Root("empty"), "color_fg") == std::string("VideoDLSS ") + DLSSVID_VERSION);
    // a long value is cut to 1 KB, a non-ASCII one replaced
    Save(root, "color_nr", PassKind::ColorNr, {{"backend", "ngx"}, {"masks", {std::string(2000, 'm')}}, {"style", 1}, {"color_source", "video"}});
    CHECK(BuildSettingsNote(root, "color_nr").size() == kSettingsNoteMax);
    nlohmann::json odd = BuildSettingsJson(root, "color_nr");
    odd["stages"]["nr"]["backend"] = "\xd0\xb1";  // «б»
    for (const char c : SettingsNoteText(odd)) CHECK((c >= 32 && c <= 126));
}

// review of PR-11: a color_fg kept while nr was recomputed under it (fg unchecked, then nr.intensity changed) - the result is made of
// the nr version process recorded in its inputs, not of the current color_nr; that version from the history, or named as gone.
TEST_CASE("Settings note follows the versions the encoded pass was made from", "[unit][encode][regression]") {
    const auto root = Root("versions");
    auto nr = [&](const char* fp, double intensity) {
        Manifest m = Manifest::ForPass(PassKind::ColorNr, 1920, 1080, FileFormat::Exr);
        m.stageParams = {{"backend", "ngx"}, {"intensity", intensity}, {"color_source", "video"}};
        m.fingerprint = std::string("sha256:") + std::string(64, fp[0]);
        m.Save(root / "color_nr");
        return m.fingerprint;
    };
    const std::string a = nr("a", 1.2);
    REQUIRE(RetirePassVersion(root, "color_nr"));
    nr("b", 1.4);
    Manifest fg = Manifest::ForPass(PassKind::ColorFg, 1920, 1080, FileFormat::Exr);
    fg.stageParams = {{"backend", "blend"}, {"multiplier", 2}, {"color_source", "pass:color_nr"}};
    fg.fingerprint = std::string("sha256:") + std::string(64, 'f');
    fg.inputs["color_nr"] = a;
    fg.Save(root / "color_fg");
    const std::string note = BuildSettingsNote(root, "color_fg");
    CHECK(note.find("nr: ngx intensity=1.2") != std::string::npos);  // before: 1.4, the current pass
    CHECK(note.find("1.4") == std::string::npos);
    CHECK(BuildSettingsNote(root, "color_nr").find("intensity=1.4") != std::string::npos);  // encoded from color_nr: the current one
    std::filesystem::remove_all(root / "color_nr.v");  // gc took it
    CHECK(BuildSettingsNote(root, "color_fg").find("nr: ? (that version is not on disk)") != std::string::npos);
}

// review round 2 of PR-11: an input recorded while the nr pass was legacy (a standalone run) - the same files adopted since (a
// fingerprint stamped), or retired into the history without one - is that version, not «not on disk»; the ratio transfer acts
// whenever the model ran at another size than the frame (an odd side at model_scale 1).
TEST_CASE("Settings note: legacy records, adopted and retired legacy versions, the work size", "[unit][encode][regression]") {
    const auto root = Root("legacy");
    Manifest legacy = Manifest::ForPass(PassKind::ColorNr, 1920, 1080, FileFormat::Exr);
    legacy.sourceHash = "sha256:5ource";
    legacy.frameCount = 48;
    legacy.stageParams = {{"backend", "ngx"}, {"intensity", 1.1}, {"color_source", "video"}};
    legacy.Save(root / "color_nr");
    const std::string rec = LegacyPassId(legacy);
    Manifest fg = Manifest::ForPass(PassKind::ColorFg, 1920, 1080, FileFormat::Exr);
    fg.stageParams = {{"backend", "blend"}, {"multiplier", 2}, {"color_source", "pass:color_nr"}};
    fg.fingerprint = std::string("sha256:") + std::string(64, 'f');
    fg.inputs["color_nr"] = rec;
    fg.Save(root / "color_fg");
    CHECK(BuildSettingsNote(root, "color_fg").find("nr: ngx intensity=1.1 |") != std::string::npos);
    // adopted: the same files with a fingerprint and stage_params.adopted
    Manifest adopted = legacy;
    adopted.fingerprint = std::string("sha256:") + std::string(64, 'a');
    adopted.stageParams["adopted"] = true;
    adopted.Save(root / "color_nr");
    CHECK(BuildSettingsNote(root, "color_fg").find("nr: ngx intensity=1.1 (adopted)") != std::string::npos);
    // retired without a fingerprint, another current pass in its place
    legacy.Save(root / "color_nr");
    REQUIRE(RetirePassVersion(root, "color_nr"));
    Manifest now = Manifest::ForPass(PassKind::ColorNr, 1920, 1080, FileFormat::Exr);
    now.stageParams = {{"backend", "ngx"}, {"intensity", 1.6}, {"color_source", "video"}};
    now.fingerprint = std::string("sha256:") + std::string(64, 'b');
    now.Save(root / "color_nr");
    const std::string note = BuildSettingsNote(root, "color_fg");
    CHECK(note.find("intensity=1.1") != std::string::npos);
    CHECK(note.find("1.6") == std::string::npos);
    CHECK(note.find("not on disk") == std::string::npos);
    // an odd side: the model at 852 for an 853-wide frame - the ratio transfer acted
    const auto odd = Root("odd");
    Manifest m = Manifest::ForPass(PassKind::ColorNr, 853, 480, FileFormat::Exr);
    m.stageParams = {{"backend", "ngx"}, {"model_scale", 1.0}, {"work_size", {852, 480}}, {"transfer", 0.5}, {"max_ratio", 6.0}, {"color_source", "video"}};
    m.Save(odd / "color_nr");
    CHECK(BuildSettingsNote(odd, "color_nr").find("model_scale=1 transfer=0.5 max_ratio=6") != std::string::npos);
    m.stageParams["work_size"] = {853, 480};
    m.Save(odd / "color_nr");
    CHECK(BuildSettingsNote(odd, "color_nr").find("transfer") == std::string::npos);
}
