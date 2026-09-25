#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <vector>

// Hygiene of the public repository: the licence is in place and wired into the package, and the documentation links
// only to public hosts — internal infrastructure (a private GitLab, a company mirror) is described with placeholders.
namespace {
const std::filesystem::path kRoot = std::filesystem::path(DLSSVID_SOURCE_DIR);

std::string ReadAll(const std::filesystem::path& p) {
    std::ifstream in(p, std::ios::binary);
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

std::vector<std::filesystem::path> DocFiles() {
    std::vector<std::filesystem::path> files = {kRoot / "README.md",
                                                kRoot / "CHANGELOG.md",
                                                kRoot / "LICENSE",
                                                kRoot / ".gitlab-ci.yml",
                                                kRoot / "app" / "README.md",
                                                kRoot / "models" / "export" / "README.md",
                                                kRoot / "sr_worker" / "README.md",
                                                kRoot / "depth_worker" / "README.md",
                                                kRoot / "bin" / "nvidia" / "README.md",
                                                kRoot / "tests" / "data" / "README.md"};
    for (const auto& e : std::filesystem::recursive_directory_iterator(kRoot / "docs"))
        if (e.is_regular_file() && e.path().extension() == ".md") files.push_back(e.path());
    return files;
}
}  // namespace

TEST_CASE("LICENSE: MIT at the repository root, shipped in the package", "[hygiene]") {
    const auto text = ReadAll(kRoot / "LICENSE");
    REQUIRE_FALSE(text.empty());
    CHECK(text.rfind("MIT License", 0) == 0);
    CHECK(text.find("Copyright (c)") != std::string::npos);
    CHECK(text.find("Permission is hereby granted, free of charge") != std::string::npos);

    const auto cmake = ReadAll(kRoot / "CMakeLists.txt");
    CHECK(cmake.find("install(FILES README.md CHANGELOG.md LICENSE DESTINATION .)") != std::string::npos);
    CHECK(cmake.find("CPACK_RESOURCE_FILE_LICENSE") != std::string::npos);

    CHECK(ReadAll(kRoot / "README.md").find("MIT License") != std::string::npos);
}

TEST_CASE("Docs link only to public hosts", "[hygiene]") {
    const std::set<std::string> allowed = {
        "github.com",          "raw.githubusercontent.com", "huggingface.co",   "arxiv.org",       "developer.nvidia.com",
        "docs.nvidia.com",     "www.nvidia.com",            "download.pytorch.org", "pytorch.org", "pypi.org",
        "cmake.org",           "vcpkg.io",                  "doc.qt.io",        "www.qt.io",       "learn.microsoft.com",
        "ffmpeg.org",          "opensource.org",            "nsis.sourceforge.io", "localhost"};
    // A placeholder host such as https://<host>/... has no host characters after :// and is not a link.
    const std::regex url(R"(https?://([A-Za-z0-9.-]+))");
    int links = 0;
    for (const auto& file : DocFiles()) {
        if (!std::filesystem::exists(file)) continue;
        const auto text = ReadAll(file);
        for (auto it = std::sregex_iterator(text.begin(), text.end(), url); it != std::sregex_iterator(); ++it) {
            const std::string host = (*it)[1].str();
            ++links;
            INFO(file.lexically_relative(kRoot).generic_string() << " links to " << host);
            CHECK(allowed.count(host) == 1);
        }
    }
    CHECK(links > 0);
}

TEST_CASE("Docs carry no developer-machine paths", "[hygiene]") {
    // SDK checkouts and user profiles of the development machine are written with placeholders (<SDK>, %LOCALAPPDATA%).
    const std::regex machine(R"re([A-Za-z]:[\\/](SDK|Users)[\\/][^\s`'")]*)re");
    std::string offenders;
    int count = 0;
    for (const auto& file : DocFiles()) {
        if (!std::filesystem::exists(file)) continue;
        const auto text = ReadAll(file);
        for (auto it = std::sregex_iterator(text.begin(), text.end(), machine); it != std::sregex_iterator(); ++it) {
            ++count;
            offenders += file.lexically_relative(kRoot).generic_string() + ": " + it->str() + "\n";
        }
    }
    INFO(offenders);
    CHECK(count == 0);
}
