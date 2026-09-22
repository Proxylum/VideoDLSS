// Regression (release 0.2.0 check): a file path longer than MAX_PATH could not be written — the TensorRT engine cache
// of a package unpacked into a deep folder was built and then lost. Win32LongPath makes such paths openable.

#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>
#include <string>

#include "util/Paths.h"

using namespace dlssvid;

TEST_CASE("Win32LongPath opens, checks and reads files beyond MAX_PATH; short paths are untouched", "[unit][paths][regression]") {
    const std::filesystem::path base = std::filesystem::path(DLSSVID_TEST_TMP) / "paths";
    // a directory chain that takes the file path past 260 characters
    std::filesystem::path deep = base;
    while (deep.string().size() < 240) deep /= "a_rather_long_directory_name_0123456789";
    // cleanup of a previous run: remove_all from a short root cannot reach children beyond MAX_PATH — the deepest
    // folder goes first through its prefixed path, the short remainder after it
    std::filesystem::remove_all(Win32LongPath(deep));
    std::filesystem::remove_all(base);
    const std::filesystem::path file = deep / "da3metric-large_1x518x924.2375828779456655.nvidia_geforce_rtx_4070_ti_super_sm89.trt10.16.1.fp16.engine";
    REQUIRE(file.string().size() > 260);

    const std::filesystem::path longFile = Win32LongPath(file);
#ifdef _WIN32
    CHECK(longFile.wstring().rfind(L"\\\\?\\", 0) == 0);
    CHECK(longFile.wstring().find(L'/') == std::wstring::npos);
    CHECK(Win32LongPath(longFile) == longFile);  // idempotent
#endif
    CHECK(Win32LongPath("relative/dir/file.bin") == std::filesystem::path("relative/dir/file.bin"));
    CHECK(Win32LongPath(base) == base);  // short: unchanged

    REQUIRE(std::filesystem::create_directories(Win32LongPath(deep)));
    {
        std::ofstream out(longFile, std::ios::binary);
        REQUIRE(out.good());
        out << "engine";
    }
    CHECK(std::filesystem::exists(longFile));
    CHECK(std::filesystem::file_size(longFile) == 6);
    std::ifstream in(longFile, std::ios::binary);
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    CHECK(text == "engine");
    in.close();
    std::filesystem::remove_all(Win32LongPath(deep));
    std::filesystem::remove_all(base);
    CHECK(!std::filesystem::exists(base));
}
