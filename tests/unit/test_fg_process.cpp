// The command line of the fg stage process (audit fg-2, TASK-0038): `dlssvid process` starts `dlssvid fg --as-stage`.
// Paths must reach the child as UTF-8 (Subprocess::Start widens from UTF-8; path.string() would be the ANSI codepage
// and silently break a Cyrillic path), the parameters as ASCII JSON, the test hooks only when set.

#include <catch2/catch_test_macros.hpp>

#include <windows.h>
#include <shellapi.h>

#include <algorithm>
#include <filesystem>
#include <string>
#include <vector>

#include "pipeline/FgProcess.h"
#include "util/Paths.h"
#include "util/Subprocess.h"

#pragma comment(lib, "shell32.lib")

using namespace dlssvid;

namespace {

std::string U8(const std::filesystem::path& p) {
    const std::u8string u = p.u8string();
    return std::string(u.begin(), u.end());
}

bool Has(const std::vector<std::string>& v, const std::string& s) { return std::find(v.begin(), v.end(), s) != v.end(); }

std::string After(const std::vector<std::string>& v, const std::string& key) {
    const auto it = std::find(v.begin(), v.end(), key);
    return it == v.end() || it + 1 == v.end() ? std::string() : *(it + 1);
}

// What the child sees: the command line Subprocess::Start builds, parsed back the way the CRT does.
std::vector<std::string> RoundTrip(const std::vector<std::string>& args) {
    std::string cmd;
    for (size_t i = 0; i < args.size(); ++i) cmd += (i ? " " : "") + QuoteArg(args[i]);
    const int n = MultiByteToWideChar(CP_UTF8, 0, cmd.c_str(), -1, nullptr, 0);
    std::wstring w(static_cast<size_t>(n - 1), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, cmd.c_str(), -1, w.data(), n);
    int argc = 0;
    LPWSTR* argv = CommandLineToArgvW(w.c_str(), &argc);
    std::vector<std::string> out;
    for (int i = 0; i < argc; ++i) out.push_back(U8(std::filesystem::path(argv[i])));
    LocalFree(argv);
    return out;
}

}  // namespace

TEST_CASE("FgProcessArgs builds the child command line: pass dirs, parameters as ASCII JSON, UTF-8 paths, hooks only when set", "[unit][process][fg][isolation]") {
    FgProcessSpec spec;
    spec.exe = std::filesystem::path(u8"C:/Программы/dlssvid.exe");
    spec.input = std::filesystem::path(u8"D:/Видео/клип.mp4");
    spec.passesRoot = std::filesystem::path(u8"D:/Видео/пассы");
    spec.colorDir = spec.passesRoot / "color_nr";
    spec.params = {{"backend", "blend"}, {"multiplier", 3}, {"note", u8"кадры \"x\" \\ y"}};
    std::vector<std::string> a = FgProcessArgs(spec);
    CHECK(a[0] == U8(spec.exe));
    CHECK(Has(a, "fg"));
    CHECK(Has(a, "--as-stage"));
    CHECK(Has(a, "--no-auto"));
    CHECK(After(a, "-i") == U8(std::filesystem::absolute(spec.input)));  // UTF-8 bytes, not the ANSI codepage
    CHECK(After(a, "-o") == U8(std::filesystem::absolute(spec.passesRoot)));
    CHECK(After(a, "--color-dir") == U8(spec.colorDir));
    CHECK_FALSE(Has(a, "--depth-dir"));  // empty folders are not passed
    CHECK_FALSE(Has(a, "--mv-dir"));
    const std::string params = After(a, "--params-json");
    CHECK(std::all_of(params.begin(), params.end(), [](char c) { return static_cast<unsigned char>(c) < 0x80; }));  // ASCII JSON
    CHECK(nlohmann::json::parse(params) == spec.params);
    for (const char* hook : {"--frames", "--warp", "--disable-unavailable", "--source-hash", "--crash-after", "--crash-on-exit", "--stall"})
        CHECK_FALSE(Has(a, hook));

    spec.frames = 12;
    spec.warp = true;
    spec.disableUnavailable = true;
    spec.sourceHash = "sha256:ab";
    spec.crashAfter = 0;
    spec.crashOnExit = true;
    spec.stall = true;
    spec.verbose = true;
    a = FgProcessArgs(spec);
    CHECK(After(a, "--frames") == "12");
    CHECK(After(a, "--source-hash") == "sha256:ab");
    CHECK(After(a, "--crash-after") == "0");
    for (const char* flag : {"--warp", "--disable-unavailable", "--crash-on-exit", "--stall", "-v"}) CHECK(Has(a, flag));
    CHECK(RoundTrip(a) == a);  // quotes, backslashes and Cyrillic survive the command line
}

TEST_CASE("DescribeExitCode names crash statuses and rejected command lines", "[unit][process][fg]") {
    CHECK(DescribeExitCode(0xC0000005u) == "0xC0000005 (access violation)");
    CHECK(DescribeExitCode(0xC00000FDu).find("stack overflow") != std::string::npos);
    CHECK(DescribeExitCode(0xC0000409u).find("0xC0000409") != std::string::npos);
    CHECK(DescribeExitCode(0xC0001234u) == "0xC0001234");
    CHECK(DescribeExitCode(3) == "3 (abort)");
    CHECK(DescribeExitCode(105).find("command line rejected") != std::string::npos);
    CHECK(DescribeExitCode(1) == "1");
}

TEST_CASE("CurrentExecutable is the running test binary", "[unit][paths]") {
    CHECK(CurrentExecutable().filename() == "dlssvid_unit_tests.exe");
    CHECK(CurrentExecutable().is_absolute());
}
