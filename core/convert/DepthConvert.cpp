#include "convert/DepthConvert.h"

#include <algorithm>
#include <cmath>

#include "util/Error.h"

namespace dlssvid {

float LinearToReverseZ(float z, float zNear, float zFar) {
    if (!(zNear > 0.f) || !(zFar > zNear)) Throw("LinearToReverseZ: need 0 < zNear < zFar");
    if (!std::isfinite(z) || z <= 0.f) return 0.f;  // unknown / behind camera -> zFar plane
    z = std::clamp(z, zNear, zFar);
    return (zNear * (zFar - z)) / (z * (zFar - zNear));
}

float ReverseZToLinear(float d, float zNear, float zFar) {
    if (!(zNear > 0.f) || !(zFar > zNear)) Throw("ReverseZToLinear: need 0 < zNear < zFar");
    if (!std::isfinite(d)) return zFar;
    d = std::clamp(d, 0.f, 1.f);
    return (zNear * zFar) / (d * (zFar - zNear) + zNear);
}

DepthRange ComputeDepthRange(const PassImage& depth) {
    DepthRange r;
    for (uint32_t y = 0; y < depth.height; ++y)
        for (uint32_t x = 0; x < depth.width; ++x) {
            const float v = depth.Get(x, y, 0);
            if (!std::isfinite(v)) continue;
            if (!r.valid) {
                r.min = r.max = v;
                r.valid = true;
            } else {
                r.min = std::min(r.min, v);
                r.max = std::max(r.max, v);
            }
        }
    return r;
}

PassImage DepthRawToDlss(const PassImage& raw, DepthParams& params) {
    if (raw.channels.size() != 1) Throw("DepthRawToDlss: expected a single-channel depth image");
    if (params.relative && params.minValue == params.maxValue) {
        const DepthRange r = ComputeDepthRange(raw);
        if (!r.valid) Throw("DepthRawToDlss: relative depth image has no finite samples");
        params.minValue = r.min;
        params.maxValue = r.max;
    }
    PassImage out = MakePassImage(PassKind::DepthDlss, raw.width, raw.height);
    const float span = params.maxValue - params.minValue;
    for (uint32_t y = 0; y < raw.height; ++y)
        for (uint32_t x = 0; x < raw.width; ++x) {
            float z = raw.Get(x, y, 0);
            if (params.relative) {
                if (!std::isfinite(z)) {
                    out.Set(x, y, 0, 0.f);
                    continue;
                }
                const float t = span > 0.f ? (z - params.minValue) / span : 0.f;
                z = params.zNear + std::clamp(t, 0.f, 1.f) * (params.zFar - params.zNear);
            }
            out.Set(x, y, 0, LinearToReverseZ(z, params.zNear, params.zFar));
        }
    return out;
}

PassImage DepthDlssToRaw(const PassImage& dlss, const DepthParams& params) {
    if (dlss.channels.size() != 1) Throw("DepthDlssToRaw: expected a single-channel depth image");
    PassImage out = MakePassImage(PassKind::DepthRaw, dlss.width, dlss.height);
    const float span = params.maxValue - params.minValue;
    for (uint32_t y = 0; y < dlss.height; ++y)
        for (uint32_t x = 0; x < dlss.width; ++x) {
            float z = ReverseZToLinear(dlss.Get(x, y, 0), params.zNear, params.zFar);
            if (params.relative) {
                const float t = (z - params.zNear) / (params.zFar - params.zNear);
                z = params.minValue + t * span;
            }
            out.Set(x, y, 0, z);
        }
    return out;
}

}  // namespace dlssvid
