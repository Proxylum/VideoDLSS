// The CLI is the contract for automation (ТЗ §7, §9): run the real executable.

#include <catch2/catch_test_macros.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>
#include <string>

#include "stages/ParamSchema.h"

#include "TestClips.h"

using namespace dlssvid;
using namespace dlssvid::test;

namespace {

int Run(const std::string& args) {
    const std::string cmd = "\"\"" DLSSVID_CLI_PATH "\" " + args + "\"";  // cmd.exe quoting
    return std::system(cmd.c_str());
}

std::string Help(const std::string& command, const std::filesystem::path& file) {
    const std::string cmd = "\"\"" DLSSVID_CLI_PATH "\" " + command + " --help > \"" + file.string() + "\"\"";
    REQUIRE(std::system(cmd.c_str()) == 0);
    std::ifstream in(file);
    std::stringstream s;
    s << in.rdbuf();
    return s.str();
}

// A value of the parameter other than its default: what makes StageCliArgs emit the option.
nlohmann::json OtherValue(const ParamSpec& p) {
    switch (p.type) {
        case ParamType::Bool: return !(p.def.is_boolean() && p.def.get<bool>());
        case ParamType::Enum:
            for (const auto& c : p.choices)
                if (nlohmann::json(c.value) != p.def) return c.value;
            return "x";
        case ParamType::Int: return static_cast<int>(p.def.is_number() && p.def.get<double>() == p.max ? p.min : p.max);
        case ParamType::Float: return p.def.is_number() && p.def.get<double>() == p.max ? p.min : p.max;
        default: return "x";
    }
}

}  // namespace

// Review of TASK-0038 (gui-runstage-depth-keys): the GUI runs a single stage as `dlssvid <stage>` with options made from the
// keys of the parameter schema; `depth --input_size` was refused, so «Запустить этап» failed for any changed depth
// setting - and so did flow's max_res / no_dlss / models_dir, upscale's jitter / target / no_fallback, nr's guides.
TEST_CASE("StageCliArgs: every schema parameter becomes an option its CLI command has", "[integration][cli][schema][regression]") {
    for (const StageSchema& s : StageSchemas()) {
        const std::string help = Help(s.stage, TempDir() / ("help_" + s.stage + ".txt"));
        for (const ParamSpec& p : s.params) {
            nlohmann::json params{{p.key, OtherValue(p)}};
            if (p.key == "target_width" || p.key == "target_height") params = {{"target_width", 1920}, {"target_height", 1080}};
            if (!p.backends.empty()) params["backend"] = p.backends.front();  // a parameter of some backends (tile, window)
            if (!p.formChoices.empty()) {  // a form list (depth.model of VDA): one of its models
                params["backend"] = p.formChoices.begin()->first;
                params[p.key] = p.formChoices.begin()->second.back().value;
            }
            const std::vector<std::string> args = StageCliArgs(s.stage, params);
            INFO(s.stage << " " << params.dump());
            int options = 0;
            for (const std::string& a : args) {
                if (a.rfind("--", 0) != 0) continue;
                ++options;
                CHECK(std::regex_search(help, std::regex(a + "([^a-z0-9_-]|$)")));
            }
            CHECK(options == (p.key != "backend" && params.contains("backend") ? 2 : 1));  // the parameter's own option (and --backend)
        }
    }
}

TEST_CASE("StageCliArgs maps defaults, cleared default-on flags, empty strings and the upscale target", "[integration][cli][schema]") {
    CHECK(StageCliArgs("upscale", {{"jitter", true}, {"no_fallback", false}, {"models_dir", ""}}).empty());  // all defaults
    CHECK(StageCliArgs("upscale", {{"jitter", false}}) == std::vector<std::string>{"--no-jitter"});
    CHECK(StageCliArgs("nr", {{"guides", false}}) == std::vector<std::string>{"--no-guides"});
    CHECK(StageCliArgs("depth", {{"no_fill", true}, {"input_size", 700}}) == std::vector<std::string>{"--input_size", "700", "--no_fill"});
    CHECK(StageCliArgs("upscale", {{"target_width", 1920}, {"target_height", 1080}, {"scale", 2}}) ==
          std::vector<std::string>{"--scale", "2", "--target", "1920x1080"});
    CHECK(StageCliArgs("upscale", {{"target_width", 1920}, {"target_height", 0}}).empty());  // 0 = by the scale
    CHECK(StageCliArgs("depth", {{"backend", "vda"}, {"model", ""}}) == std::vector<std::string>{"--backend", "vda"});
    // TASK-0039: a window left after a switch to DA3, and the default window, are not passed
    CHECK(StageCliArgs("depth", {{"backend", "da3"}, {"window", 16}, {"model", ""}}) == std::vector<std::string>{"--backend", "da3"});
    CHECK(StageCliArgs("depth", {{"backend", "vda"}, {"window", 0}}) == std::vector<std::string>{"--backend", "vda"});
    CHECK(StageCliArgs("depth", {{"backend", "vda"}, {"window", 16}, {"model", "metric-vda-large"}}) ==
          std::vector<std::string>{"--backend", "vda", "--model", "metric-vda-large", "--window", "16"});
}

TEST_CASE("dlssvid --version and info run", "[integration][cli]") {
    CHECK(Run("--version") == 0);
    ClipSpec spec;
    spec.frames = 3;
    const auto clip = WriteClip(TempDir() / "cli_info.mkv", spec);
    CHECK(Run("info --warp \"" + clip.string() + "\"") == 0);
}

TEST_CASE("dlssvid process --passthrough --codec ffv1 is lossless", "[integration][cli]") {
    ClipSpec spec;
    spec.frames = 10;
    spec.audio = true;
    const auto in = WriteClip(TempDir() / "cli_in.mkv", spec);
    const auto out = TempDir() / "cli_out.mkv";
    std::filesystem::remove(out);

    const int rc = Run("process --passthrough --codec ffv1 --warp -i \"" + in.string() + "\" -o \"" + out.string() + "\"");
    REQUIRE(rc == 0);
    REQUIRE(std::filesystem::exists(out));

    const auto a = DecodeAll(in), b = DecodeAll(out);
    REQUIRE(a.size() == b.size());
    for (size_t i = 0; i < a.size(); ++i) REQUIRE(a[i].data == b[i].data);
    CHECK(Summarize(out).audioStreams == 1);
}

TEST_CASE("dlssvid process --frames limits the run", "[integration][cli]") {
    ClipSpec spec;
    spec.frames = 10;
    const auto in = WriteClip(TempDir() / "cli_frames_in.mkv", spec);
    const auto out = TempDir() / "cli_frames_out.mkv";
    REQUIRE(Run("process --passthrough --codec ffv1 --warp --frames 4 -i \"" + in.string() + "\" -o \"" + out.string() + "\"") == 0);
    CHECK(DecodeAll(out).size() == 4);
}

TEST_CASE("dlssvid process rejects an unknown stage (the full pipeline arrived in stage 8)", "[integration][cli]") {
    ClipSpec spec;
    spec.frames = 2;
    const auto in = WriteClip(TempDir() / "cli_reject_in.mkv", spec);
    CHECK(Run("process --codec ffv1 --warp --stages bogus -i \"" + in.string() + "\" -o \"" + (TempDir() / "cli_reject_out.mkv").string() + "\"") != 0);
}
