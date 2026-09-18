#include "util/Sha256.h"

#include <windows.h>
#include <bcrypt.h>

#include <fstream>
#include <vector>

#include "util/Error.h"

#pragma comment(lib, "bcrypt.lib")

namespace dlssvid {

namespace {

class Sha256 {
public:
    Sha256() {
        if (!BCRYPT_SUCCESS(BCryptOpenAlgorithmProvider(&alg_, BCRYPT_SHA256_ALGORITHM, nullptr, 0)))
            Throw("BCryptOpenAlgorithmProvider(SHA256) failed");
        DWORD objLen = 0, cb = 0;
        BCryptGetProperty(alg_, BCRYPT_OBJECT_LENGTH, reinterpret_cast<PUCHAR>(&objLen), sizeof(objLen), &cb, 0);
        object_.resize(objLen);
        if (!BCRYPT_SUCCESS(BCryptCreateHash(alg_, &hash_, object_.data(), objLen, nullptr, 0, 0)))
            Throw("BCryptCreateHash failed");
    }
    ~Sha256() {
        if (hash_) BCryptDestroyHash(hash_);
        if (alg_) BCryptCloseAlgorithmProvider(alg_, 0);
    }
    void Update(const void* data, size_t size) {
        if (!BCRYPT_SUCCESS(BCryptHashData(hash_, static_cast<PUCHAR>(const_cast<void*>(data)), static_cast<ULONG>(size), 0)))
            Throw("BCryptHashData failed");
    }
    std::string Finish() {
        unsigned char digest[32];
        if (!BCRYPT_SUCCESS(BCryptFinishHash(hash_, digest, sizeof(digest), 0))) Throw("BCryptFinishHash failed");
        static const char* hex = "0123456789abcdef";
        std::string out(64, '0');
        for (int i = 0; i < 32; ++i) {
            out[2 * i] = hex[digest[i] >> 4];
            out[2 * i + 1] = hex[digest[i] & 15];
        }
        return out;
    }

private:
    BCRYPT_ALG_HANDLE alg_ = nullptr;
    BCRYPT_HASH_HANDLE hash_ = nullptr;
    std::vector<UCHAR> object_;
};

}  // namespace

std::string Sha256Hex(const void* data, size_t size) {
    Sha256 h;
    h.Update(data, size);
    return h.Finish();
}

std::string Sha256File(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) Throw("cannot open for hashing: " + path.string());
    Sha256 h;
    std::vector<char> buf(1 << 20);
    while (in) {
        in.read(buf.data(), static_cast<std::streamsize>(buf.size()));
        const std::streamsize n = in.gcount();
        if (n > 0) h.Update(buf.data(), static_cast<size_t>(n));
    }
    return h.Finish();
}

}  // namespace dlssvid
