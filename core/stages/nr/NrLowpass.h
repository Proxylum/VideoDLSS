#pragma once

#include <cstdint>

#include "passes/PassImage.h"

namespace dlssvid {

// The low band of the NR edit (PR-8, the G / B split of the control mask; HACK in docs/architecture.md): the means of d x d cells over
// the pixels that exist (an edge cell has fewer; non-finite samples count as 0) -> Gauss [1 2 1] x [1 2 1] / 16 on the grid, edges
// clamped -> bilinear back to the frame in fp32, cell i centred at (i + 0.5) d, edges clamped. The CPU twin of CSEditBox, CSEditGauss
// and LowUp of NrCompose.hlsl in the same order of float operations: the kernel tests, `compare --bands` and E10 measure with it.

// Cells across the short side of the output frame: d is one share of the picture's height at any resolution (E10, docs/benchmarks.md).
inline constexpr uint32_t kNrLowpassCells = 180;
// Appended to tool.app of an nr pass that has mask_control as an input (ProcessRunner): a later change of kNrLowpassCells or of the
// mask variant's code gives it a value ("+nr.cmask.2") and recomputes only the passes made with a mask. Empty in PR-8.
inline constexpr const char* kNrControlMaskRevision = "";

// d for a w x h frame: max(2, round(min(w, h) / kNrLowpassCells)); a non-zero override wins.
uint32_t NrLowpassDivisor(uint32_t w, uint32_t h, uint32_t override = 0);

struct NrLowGrid {
    uint32_t width = 0, height = 0, div = 0;  // ceil(w / d) x ceil(h / d)
};
NrLowGrid NrLowpassGrid(uint32_t w, uint32_t h, uint32_t div);

// The low band of e (any sample type and channel count, values as they are: no normalisation) at d: F32, the size and channels of e.
// = NrLowpassUpCpu(NrLowBandCpu(e, d), e.width, e.height, d).
PassImage NrLowpassCpu(const PassImage& e, uint32_t div);
// Its two halves: the grid after the box means and the Gauss (what CSEditGauss writes, NrLowpassGrid's size, F32) and the bilinear
// way back to a w x h frame (LowUp).
PassImage NrLowBandCpu(const PassImage& e, uint32_t div);
PassImage NrLowpassUpCpu(const PassImage& grid, uint32_t w, uint32_t h, uint32_t div);
// The share of the energy of diff in its low band: sum |LowPass_d(diff)|^2 / sum |diff|^2 over every sample; 0 for a zero diff.
double NrLowShare(const PassImage& diff, uint32_t div);

}  // namespace dlssvid
