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

// TASK-0066: every executable of the project runs with UTF-8 as its process code page (cmake/utf8.manifest), so a
// std::string path is UTF-8 both ways. Under the ANSI code page (CP1251) path::string() of a Cyrillic folder was not
// UTF-8: the CLI aborted on such an argument, a project naming such a file could not be written as JSON.
TEST_CASE("The process code page is UTF-8: a std::string path is UTF-8 both ways", "[unit][paths][regression]") {
    REQUIRE(Utf8CodePage());
    const std::u8string name = u8"Видео проекта";
    const std::string utf8(name.begin(), name.end());
    CHECK(std::filesystem::path(utf8) == std::filesystem::path(name));
    CHECK(std::filesystem::path(name).string() == utf8);
    const std::filesystem::path dir = std::filesystem::path(DLSSVID_TEST_TMP) / "paths_utf8" / name;
    std::filesystem::remove_all(dir.parent_path());
    std::filesystem::create_directories(dir);
    std::ofstream(dir.string() + "/файл.txt") << "ok";  // a narrow file name, as the code builds them
    CHECK(std::filesystem::exists(dir / u8"файл.txt"));
}

// Review of TASK-0066: on a Windows that ignores the manifest the CLI refuses a non-ASCII argument. It read the bytes of
// the ANSI argv first, where a letter outside the code page is already a best-fit ASCII one or '?' - D:\Café passed
// as D:\Cafe. The wide command line shows every such argument; the program's own path does not count.
TEST_CASE("FirstNonAsciiArgument reads the wide command line, skips the program's path", "[unit][paths][regression]") {
    CHECK(FirstNonAsciiArgument(L"dlssvid.exe info D:\\clips\\a.mp4") == 0);
    CHECK(FirstNonAsciiArgument(L"dlssvid.exe process -o \"D:\\Caf\u00E9\\out.mkv\"") == 3);  // é: best-fit e in CP1251
    CHECK(FirstNonAsciiArgument(L"dlssvid.exe info D:\\\u6771\u4EAC\\a.mp4") == 2);           // 東京: '?' in CP1251
    CHECK(FirstNonAsciiArgument(L"dlssvid.exe info D:\\\u0412\u0438\u0434\u0435\u043E\\a.mp4 -v") == 2);
    CHECK(FirstNonAsciiArgument(L"\"C:\\\u041F\u0440\u043E\u0433\u0440\u0430\u043C\u043C\u044B\\dlssvid.exe\" info a.mp4") == 0);
    CHECK(FirstNonAsciiArgument(L"") == 0);
    CHECK(FirstNonAsciiArgument() == 0);  // this test's own command line: ASCII
}
