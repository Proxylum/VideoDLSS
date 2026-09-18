#pragma once

#include "passes/Manifest.h"
#include "passes/PassImage.h"

namespace dlssvid {

// docs/conventions.md §2. Linear (view-space, metres, larger = farther) <-> DLSS reverse-Z
// depth in [0,1] with 1 = near plane, 0 = far plane.
float LinearToReverseZ(float z, float zNear, float zFar);
float ReverseZToLinear(float d, float zNear, float zFar);

struct DepthRange {
    float min = 0.f;
    float max = 0.f;
    bool valid = false;
};
DepthRange ComputeDepthRange(const PassImage& depth);  // ignores non-finite samples

// depth_raw -> depth_dlss. For relative depth (params.relative) the raw values are mapped
// linearly onto [near, far] using params.minValue/maxValue; when those are 0 they are
// computed from the image and written back into params so the manifest stays reversible.
PassImage DepthRawToDlss(const PassImage& raw, DepthParams& params);

// depth_dlss -> depth_raw (metres, or relative values when params.relative).
PassImage DepthDlssToRaw(const PassImage& dlss, const DepthParams& params);

}  // namespace dlssvid
