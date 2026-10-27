#pragma once

#include <cstdint>

namespace dlssvid {

// The control mask of NR (PR-8, docs/conventions.md §6): one static RGB image over every frame, a pass of one frame 0 - R the share of
// the edit, G / B the weights of its low / high frequencies (a grey image is one channel: the edit x v). Imported once
// (pipeline/ControlMask: `dlssvid import --pass mask_control --static`, the GUI's «Загрузить…»), read by NR, the plan and the viewport.
inline constexpr const char* kControlMaskPass = "mask_control";
inline constexpr uint32_t kControlMaskMaxSide = 16384;          // a D3D12 texture side: the viewport loads the mask as it is
inline constexpr uint64_t kControlMaskMaxPixels = 1ull << 26;   // ~67 Mpx: what decoding may take in the GUI process

}  // namespace dlssvid
