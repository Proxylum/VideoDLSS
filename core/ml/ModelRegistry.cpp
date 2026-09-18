#include "ml/ModelRegistry.h"

#include <windows.h>
#include <winhttp.h>

#include <cstdlib>
#include <fstream>

#include "util/Error.h"
#include "util/Log.h"
#include "util/Sha256.h"

#pragma comment(lib, "winhttp.lib")

namespace dlssvid {

namespace {

std::filesystem::path ExeDir() {
    wchar_t buf[MAX_PATH];
    const DWORD n = GetModuleFileNameW(nullptr, buf, MAX_PATH);
    return n ? std::filesystem::path(buf).parent_path() : std::filesystem::path();
}

std::wstring Widen(const std::string& s) {
    if (s.empty()) return {};
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
    std::wstring w(static_cast<size_t>(n > 0 ? n - 1 : 0), L'\0');
    if (n > 1) MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, w.data(), n);
    return w;
}

struct HInternet {
    HINTERNET h = nullptr;
    ~HInternet() {
        if (h) WinHttpCloseHandle(h);
    }
};

}  // namespace

std::filesystem::path ModelRegistry::DefaultRegistryPath() {
    if (const char* e = std::getenv("DLSSVID_MODELS_DIR")) return std::filesystem::path(e) / "registry.json";
    const auto exe = ExeDir();
    for (const auto& base : {exe, exe.parent_path(), exe.parent_path().parent_path(), exe.parent_path().parent_path().parent_path()}) {
        if (base.empty()) continue;
        const auto p = base / "models" / "registry.json";
        if (std::filesystem::exists(p)) return p;
    }
    return std::filesystem::current_path() / "models" / "registry.json";
}

ModelRegistry ModelRegistry::Load(const std::filesystem::path& registryJson, const std::filesystem::path& cacheDir) {
    std::ifstream in(registryJson, std::ios::binary);
    if (!in) Throw("model registry not found: " + registryJson.string());
    nlohmann::json j;
    try {
        in >> j;
    } catch (const std::exception& e) {
        Throw("registry.json parse error: " + std::string(e.what()));
    }
    ModelRegistry r;
    r.cacheDir_ = cacheDir.empty() ? registryJson.parent_path() / "cache" : cacheDir;
    for (const auto& m : j.value("models", nlohmann::json::array())) {
        ModelEntry e;
        e.id = m.value("id", "");
        e.stage = m.value("stage", "");
        e.role = m.value("role", "");
        e.source = m.value("source", "");
        e.format = m.value("format", "");
        e.url = m.value("url", "");
        e.sha256 = m.value("sha256", "");
        e.license = m.value("license", "");
        e.fileName = m.value("file", "");
        e.params = m.value("params", nlohmann::json::object());
        if (e.id.empty()) continue;
        if (e.fileName.empty()) {
            if (!e.url.empty()) {
                const size_t q = e.url.find('?');
                const std::string path = q == std::string::npos ? e.url : e.url.substr(0, q);
                e.fileName = path.substr(path.rfind('/') + 1);
            } else {
                e.fileName = e.id + (e.format == "onnx" ? ".onnx" : e.format == "safetensors" ? ".safetensors" : ".bin");
            }
        }
        r.entries_.push_back(std::move(e));
    }
    return r;
}

const ModelEntry* ModelRegistry::Find(const std::string& id) const {
    for (const auto& e : entries_)
        if (e.id == id) return &e;
    return nullptr;
}

const ModelEntry& ModelRegistry::Get(const std::string& id) const {
    const ModelEntry* e = Find(id);
    if (!e) {
        std::string known;
        for (const auto& m : entries_) known += (known.empty() ? "" : ", ") + m.id;
        Throw("unknown model '" + id + "' (registry has: " + known + ")");
    }
    return *e;
}

std::filesystem::path ModelRegistry::LocalPath(const ModelEntry& e) const { return cacheDir_ / e.fileName; }

bool ModelRegistry::IsCached(const ModelEntry& e) const { return std::filesystem::exists(LocalPath(e)); }

std::filesystem::path ModelRegistry::Fetch(const ModelEntry& e, const ProgressFn& progress) const {
    const auto path = LocalPath(e);
    if (!std::filesystem::exists(path)) {
        if (e.url.empty())
            Throw("model '" + e.id + "' is not in the cache (" + path.string() + ") and has no download URL: produce it with models/export (see models/export/README.md)");
        std::filesystem::create_directories(cacheDir_);
        const auto tmp = path.string() + ".part";
        Log()->info("downloading {} -> {}", e.url, path.string());
        DownloadFile(e.url, tmp, progress);
        std::filesystem::rename(tmp, path);
    }
    if (!e.sha256.empty()) {
        const std::string actual = Sha256File(path);
        if (actual != e.sha256) Throw("SHA-256 mismatch for " + path.string() + ": expected " + e.sha256 + ", got " + actual);
    }
    return path;
}

void DownloadFile(const std::string& url, const std::filesystem::path& dest, const ModelRegistry::ProgressFn& progress) {
    const std::wstring wurl = Widen(url);
    URL_COMPONENTS uc{};
    uc.dwStructSize = sizeof(uc);
    wchar_t host[256] = {}, path[2048] = {};
    uc.lpszHostName = host;
    uc.dwHostNameLength = 256;
    uc.lpszUrlPath = path;
    uc.dwUrlPathLength = 2048;
    if (!WinHttpCrackUrl(wurl.c_str(), 0, 0, &uc)) Throw("invalid URL: " + url);

    HInternet session{WinHttpOpen(L"dlssvid/0.1", WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY, WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0)};
    if (!session.h) Throw("WinHttpOpen failed");
    HInternet conn{WinHttpConnect(session.h, host, uc.nPort, 0)};
    if (!conn.h) Throw("WinHttpConnect failed for " + url);
    const DWORD flags = uc.nScheme == INTERNET_SCHEME_HTTPS ? WINHTTP_FLAG_SECURE : 0;
    HInternet req{WinHttpOpenRequest(conn.h, L"GET", path, nullptr, WINHTTP_NO_REFERER, WINHTTP_DEFAULT_ACCEPT_TYPES, flags)};
    if (!req.h) Throw("WinHttpOpenRequest failed for " + url);
    DWORD redirect = WINHTTP_OPTION_REDIRECT_POLICY_ALWAYS;
    WinHttpSetOption(req.h, WINHTTP_OPTION_REDIRECT_POLICY, &redirect, sizeof(redirect));
    if (!WinHttpSendRequest(req.h, WINHTTP_NO_ADDITIONAL_HEADERS, 0, WINHTTP_NO_REQUEST_DATA, 0, 0, 0) || !WinHttpReceiveResponse(req.h, nullptr))
        Throw("HTTP request failed for " + url + " (error " + std::to_string(GetLastError()) + ")");

    DWORD status = 0, size = sizeof(status);
    WinHttpQueryHeaders(req.h, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER, WINHTTP_HEADER_NAME_BY_INDEX, &status, &size, WINHTTP_NO_HEADER_INDEX);
    if (status != 200) Throw("HTTP " + std::to_string(status) + " for " + url);
    uint64_t total = 0;
    {
        wchar_t len[32] = {};
        DWORD lenSize = sizeof(len);
        if (WinHttpQueryHeaders(req.h, WINHTTP_QUERY_CONTENT_LENGTH, WINHTTP_HEADER_NAME_BY_INDEX, len, &lenSize, WINHTTP_NO_HEADER_INDEX))
            total = std::wcstoull(len, nullptr, 10);
    }

    std::ofstream out(dest, std::ios::binary);
    if (!out) Throw("cannot write " + dest.string());
    std::vector<char> buf(1 << 20);
    uint64_t done = 0;
    for (;;) {
        DWORD avail = 0;
        if (!WinHttpQueryDataAvailable(req.h, &avail)) Throw("WinHttpQueryDataAvailable failed");
        if (avail == 0) break;
        DWORD read = 0;
        if (!WinHttpReadData(req.h, buf.data(), static_cast<DWORD>(std::min<size_t>(buf.size(), avail)), &read)) Throw("WinHttpReadData failed");
        out.write(buf.data(), read);
        done += read;
        if (progress) progress(done, total);
    }
    if (total && done != total) Throw("download truncated: " + std::to_string(done) + " of " + std::to_string(total) + " bytes for " + url);
}

}  // namespace dlssvid
