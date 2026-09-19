#include "stages/fg/BlendFrameGenerator.h"

#include <algorithm>

#include "util/Error.h"

namespace dlssvid {

PassImage ToRgbF32(const PassImage& img) {
    if (img.Empty() || img.channels.size() < 3) Throw("fg: an RGB image is required");
    PassImage out;
    out.Allocate(img.width, img.height, PixelType::F32, {"R", "G", "B"});
    const float scale = img.type == PixelType::U8 ? 1.f / 255.f : img.type == PixelType::U16 ? 1.f / 65535.f : 1.f;
    for (uint32_t y = 0; y < img.height; ++y)
        for (uint32_t x = 0; x < img.width; ++x)
            for (size_t c = 0; c < 3; ++c) out.Set(x, y, c, img.Get(x, y, c) * scale);
    return out;
}

void BlendFrameGenerator::Init(D3D12Device& device, const FgConfig& config) {
    config_ = config;
    calls_ = 0;
    diag_ = {};
    diag_.gpu = device.AdapterName();
    diag_.architecture = GpuArchitectureFromName(device.AdapterName());
    diag_.driver = NvidiaDriverFromUmd(device.UmdDriverVersion());
    diag_.driverOk = diag_.driver.AtLeast(kNrMinDriverMajor, kNrMinDriverMinor);
    diag_.available = true;
    diag_.multiFrameMax = 3;
    diag_.createResult = "blend";
    diag_.ok = true;
}

void BlendFrameGenerator::Generate(const FgInputs& in, std::vector<PassImage>& out) {
    if (!in.prev.cpu || !in.cur.cpu) Throw("blend fg: CPU images of both frames are required");
    const PassImage a = ToRgbF32(*in.prev.cpu), b = ToRgbF32(*in.cur.cpu);
    if (a.width != b.width || a.height != b.height) Throw("blend fg: frame sizes differ");
    out.clear();
    for (int k = 1; k < config_.multiplier; ++k) {
        const float t = static_cast<float>(k) / static_cast<float>(config_.multiplier);
        PassImage img = MakePassImage(PassKind::ColorFg, a.width, a.height, PixelType::F16);
        for (uint32_t y = 0; y < a.height; ++y)
            for (uint32_t x = 0; x < a.width; ++x)
                for (size_t c = 0; c < 3; ++c) img.Set(x, y, c, std::clamp(a.Get(x, y, c) * (1.f - t) + b.Get(x, y, c) * t, 0.f, 1.f));
        out.push_back(std::move(img));
    }
    ++calls_;
    lastReset_ = in.reset;
}

nlohmann::json BlendFrameGenerator::Describe() const { return {{"backend", "blend"}, {"multiplier", config_.multiplier}, {"guides", false}}; }

}  // namespace dlssvid
