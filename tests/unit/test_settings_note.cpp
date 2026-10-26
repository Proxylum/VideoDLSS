#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <string>

#include "passes/Manifest.h"
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
          {"auto_mask", false}, {"guides", true}, {"scene_cut_threshold", 0.24}, {"scene_cut_count", 3}, {"scene_cuts", {40, 97, 130}}, {"temporal", 0.0},
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
                      " | upscale: dlss x2 quality=DLAA | nr: ngx style=natural intensity=1.2 passes=1 local_tone=1 local_structure=1 skin_structure=follow"
                      " auto_mask=off guides=on scene_cut=0.24 cuts=3 temporal=0 model_scale=1 | fg: dlssg x2");
    CHECK(note.find(":/") == std::string::npos);  // no C:/..., D:/...
    CHECK(note.find("nvngx") == std::string::npos);
    CHECK(note.size() <= kSettingsNoteMax);
    for (const char c : note) CHECK((c >= 32 && c <= 126));
    const nlohmann::json j = BuildSettingsJson(root, "color_fg");
    CHECK(j["stages"]["nr"]["cuts"] == 3);
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
        m.inputs["color_nr"] = "sha256:00";
        m.fingerprint = "sha256:01";  // inputs are written with the fingerprint (a pass process stamped)
        m.Save(root / "color_fg");
        CHECK(BuildSettingsNote(root, "color_fg").find("| nr: stub | fg: blend x2") != std::string::npos);
    }
    // the bypass (intensity 0, PR-7): no model
    Save(root, "color_nr", PassKind::ColorNr, {{"backend", "ngx"}, {"bypass", true}, {"passes", 0}, {"intensity", 0.0}, {"color_source", "color_sr"}});
    CHECK(BuildSettingsNote(root, "color_nr").find("nr: ngx (no model) intensity=0") != std::string::npos);
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
