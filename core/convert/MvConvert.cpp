#include "convert/MvConvert.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <vector>

#include "util/Error.h"

namespace dlssvid {

namespace {

void CheckMv(const PassImage& mv, const char* what) {
    if (mv.channels.size() != 2) Throw(std::string(what) + ": expected a 2-channel (u, v) image");
}

}  // namespace

PassImage ZeroMv(uint32_t width, uint32_t height) { return MakePassImage(PassKind::MvDlss, width, height); }

PassImage InvertFlow(const PassImage& flow, const PassImage* depth, bool fillHoles, PassImage* zbufOut) {
    CheckMv(flow, "InvertFlow");
    if (depth && (depth->width != flow.width || depth->height != flow.height)) Throw("InvertFlow: depth/flow size mismatch");
    const uint32_t w = flow.width, h = flow.height;
    PassImage out = MakePassImage(PassKind::MvDlss, w, h);
    std::vector<float> zbuf(static_cast<size_t>(w) * h, std::numeric_limits<float>::infinity());
    std::vector<uint8_t> filled(static_cast<size_t>(w) * h, 0);
    float* o = out.As<float>();

    // Forward splat: every source pixel lands on round(p + F(p)).
    for (uint32_t y = 0; y < h; ++y)
        for (uint32_t x = 0; x < w; ++x) {
            const float u = flow.Get(x, y, 0), v = flow.Get(x, y, 1);
            if (!std::isfinite(u) || !std::isfinite(v)) continue;
            const long tx = std::lround(static_cast<float>(x) + u), ty = std::lround(static_cast<float>(y) + v);
            if (tx < 0 || ty < 0 || tx >= static_cast<long>(w) || ty >= static_cast<long>(h)) continue;
            const size_t ti = static_cast<size_t>(ty) * w + static_cast<size_t>(tx);
            const float z = depth ? depth->Get(x, y, 0) : 0.f;
            const float zc = std::isfinite(z) ? z : std::numeric_limits<float>::max();
            if (filled[ti] && depth && zc >= zbuf[ti]) continue;  // an already-splatted closer surface wins
            o[2 * ti] = -u;
            o[2 * ti + 1] = -v;
            zbuf[ti] = depth ? zc : 0.f;
            filled[ti] = 1;
        }

    if (fillHoles) {
        // Multi-source BFS from filled pixels: each hole takes the vector of its nearest filled neighbour.
        std::queue<size_t> q;
        for (size_t i = 0; i < filled.size(); ++i)
            if (filled[i]) q.push(i);
        const int dx[4] = {1, -1, 0, 0}, dy[4] = {0, 0, 1, -1};
        while (!q.empty()) {
            const size_t i = q.front();
            q.pop();
            const long x = static_cast<long>(i % w), y = static_cast<long>(i / w);
            for (int k = 0; k < 4; ++k) {
                const long nx = x + dx[k], ny = y + dy[k];
                if (nx < 0 || ny < 0 || nx >= static_cast<long>(w) || ny >= static_cast<long>(h)) continue;
                const size_t ni = static_cast<size_t>(ny) * w + static_cast<size_t>(nx);
                if (filled[ni]) continue;
                o[2 * ni] = o[2 * i];
                o[2 * ni + 1] = o[2 * i + 1];
                zbuf[ni] = zbuf[i];
                filled[ni] = 1;
                q.push(ni);
            }
        }
    }
    if (zbufOut) {
        *zbufOut = MakePassImage(PassKind::DepthRaw, w, h);
        float* z = zbufOut->As<float>();
        for (size_t i = 0; i < zbuf.size(); ++i) z[i] = filled[i] ? zbuf[i] : std::numeric_limits<float>::infinity();
    }
    return out;
}

namespace {

PassImage ConvertWithOptions(const PassImage& field, const PassImage* depth, const MvConvertOptions& opt) {
    PassImage zbuf;
    PassImage inv = InvertFlow(field, depth, opt.fillHoles, depth ? &zbuf : nullptr);
    if (depth && opt.dilateRadius > 0) inv = DilateMvByDepth(inv, zbuf, opt.dilateRadius, opt.depthEdgeThreshold);
    if (opt.targetWidth && opt.targetHeight && (opt.targetWidth != inv.width || opt.targetHeight != inv.height))
        inv = ScaleMv(inv, opt.targetWidth, opt.targetHeight);
    return inv;
}

}  // namespace

PassImage ForwardFlowToBackwardMv(const PassImage& flowT, const PassImage* depthT, const MvConvertOptions& opt) {
    return ConvertWithOptions(flowT, depthT, opt);
}

PassImage BackwardMvToForwardFlow(const PassImage& mvT, const PassImage* depthT, const MvConvertOptions& opt) {
    PassImage out = ConvertWithOptions(mvT, depthT, opt);
    out.channels = Spec(PassKind::MvRaw).channels;
    return out;
}

PassImage ScaleMv(const PassImage& mv, uint32_t tw, uint32_t th) {
    CheckMv(mv, "ScaleMv");
    if (tw == 0 || th == 0) Throw("ScaleMv: zero target size");
    PassImage out;
    out.Allocate(tw, th, PixelType::F32, mv.channels);
    const float sx = static_cast<float>(tw) / mv.width, sy = static_cast<float>(th) / mv.height;
    for (uint32_t y = 0; y < th; ++y) {
        const uint32_t syi = std::min(mv.height - 1, static_cast<uint32_t>((y + 0.5f) / sy));
        for (uint32_t x = 0; x < tw; ++x) {
            const uint32_t sxi = std::min(mv.width - 1, static_cast<uint32_t>((x + 0.5f) / sx));
            out.Set(x, y, 0, mv.Get(sxi, syi, 0) * sx);
            out.Set(x, y, 1, mv.Get(sxi, syi, 1) * sy);
        }
    }
    return out;
}

PassImage DilateMvByDepth(const PassImage& mv, const PassImage& depth, int radius, float threshold) {
    CheckMv(mv, "DilateMvByDepth");
    if (depth.width != mv.width || depth.height != mv.height) Throw("DilateMvByDepth: size mismatch");
    if (radius <= 0) return mv;
    PassImage out = mv.type == PixelType::F32 ? mv : mv.ConvertTo(PixelType::F32);
    const long w = mv.width, h = mv.height;
    for (long y = 0; y < h; ++y)
        for (long x = 0; x < w; ++x) {
            const float z0 = depth.Get(static_cast<uint32_t>(x), static_cast<uint32_t>(y), 0);
            float best = z0;
            long bx = x, by = y;
            for (long dy = -radius; dy <= radius; ++dy)
                for (long dx = -radius; dx <= radius; ++dx) {
                    const long nx = x + dx, ny = y + dy;
                    if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
                    const float z = depth.Get(static_cast<uint32_t>(nx), static_cast<uint32_t>(ny), 0);
                    if (std::isfinite(z) && z < best) {
                        best = z;
                        bx = nx;
                        by = ny;
                    }
                }
            const bool edge = std::isfinite(z0) ? (z0 - best) > threshold * std::max(std::fabs(z0), 1e-6f) : std::isfinite(best);
            if (edge && (bx != x || by != y)) {
                out.Set(static_cast<uint32_t>(x), static_cast<uint32_t>(y), 0, mv.Get(static_cast<uint32_t>(bx), static_cast<uint32_t>(by), 0));
                out.Set(static_cast<uint32_t>(x), static_cast<uint32_t>(y), 1, mv.Get(static_cast<uint32_t>(bx), static_cast<uint32_t>(by), 1));
            }
        }
    return out;
}

PassImage FlipMvY(const PassImage& mv) {
    CheckMv(mv, "FlipMvY");
    PassImage out = mv;
    for (uint32_t y = 0; y < mv.height; ++y)
        for (uint32_t x = 0; x < mv.width; ++x) out.Set(x, y, 1, -mv.Get(x, y, 1));
    return out;
}

}  // namespace dlssvid
