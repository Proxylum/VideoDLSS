#pragma once

#include <cstdint>

// TASK-0080 (HDR step TC): the plain view of the tables one row of an HDR source goes through on its way to SDR (tc-plan §15.5). The
// portable row (HdrTransfer::Apply, HdrGamut::Map) and the AVX2 row (HdrRowAvx2.cpp) read the same numbers from these structs, so both
// give the same bits. This header is the only project header HdrRowAvx2.cpp includes: it has no inline functions and no templates -
// an inline function compiled with /arch:AVX2 could be the copy the linker keeps for every caller, also on a CPU without AVX2.
namespace dlssvid::hdrrow {

// The pseudo-hue axis of a cube's cusp: 6 vertices of its ring (pa relative to pa0), the columns of each gap, I of the ring point per
// column.
struct Axis {
    float pa0;
    float vertex[6];
    float scale[6];  // columns per unit of pa in each gap
    int32_t start[6];
    int32_t n;
    const float* cuspI;  // [n]
};

struct Bounds709 {  // B709(I, pa): 128 rows, 12 knots per row (vertices and edge knots), columns between knots
    Axis axis;
    const float* knot;  // [128][12]
    float cols[12];
    int32_t start[12];
    int32_t n;
    const float* node;  // [128][n]
};

struct Bounds2020 {  // B2020(I, pa): 128 rows on the cusp axis's columns
    Axis axis;
    const float* node;  // [128][axis.n]
};

// The gamut step G1 of `sdr1`: linear BT.2020 after the EETF -> linear BT.709 in [0, 1].
struct Step {
    float t[9];       // BT.2020 -> BT.709
    float w709[3];    // Y of BT.709: the box measure of level 1
    float mlms[9];    // linear BT.2020 -> relative LMS
    float ict[6];     // the Ct and Cp rows of L'M'S' -> ICtCp
    float ictInv[6];  // columns 1 and 2 of ICtCp -> L'M'S' for L', M', S' (column 0 is 1)
    float a709[9];    // relative LMS -> linear BT.709
    float knee, kneeRest, tau;  // k, 1 - k, the level-1 threshold
    float uc, ucRest;           // the cusp row's u (95/127) and 1 - u
    float e0, iw;               // I of no light and of SDR white (203 nits)
    float decScale;             // 4096 / iw
    const float* enc;           // PQ inverse of relative LMS by the float's bits: 32 octaves x 256 + 2
    const float* dec;           // PQ EOTF -> relative LMS on [0, iw]: 4096 + 2
    Bounds709 b709;
    Bounds2020 b2020;
};

// The transfer of one curve and gamut: R'G'B' codes -> linear light after the EETF (1.0 = 203 nits) -> G1 -> the SDR signal.
struct Chain {
    int32_t hlg;          // 0: PQ, 1: HLG
    const float* eotf;    // PQ: code -> relative light; HLG: code -> scene light. kPoints + 1
    const float* gain;    // the EETF's gain by maxRGB: PQ by the code, HLG by the relative display light * gainScale. kPoints + 1
    const float* ys02;    // HLG: Ys^0.2 by the float's bits, 32 octaves x 256 + 2
    const float* oetf;    // sqrt(L) -> the BT.1886 inverse. kPoints + 1
    float w[3];           // HLG: the source primaries' Y weights
    float hlgScale;       // HLG: 1000 / 203
    float gainScale;      // HLG: 203 / 1000
    const Step* gamut;    // nullptr: the source is BT.709, no gamut step
};

// Y'CbCr -> R'G'B' of a row (the numbers of ColorConvert's YuvCoeffs).
struct Yuv {
    float yOff, yScale, cOff, cScale, crR, cbB, cbG, crG;
};

// True when the CPU and the OS run AVX2 and the rest of the set /arch:AVX2 lets MSVC use (FMA3, BMI1, BMI2, LZCNT).
bool HasAvx2();

// The AVX2 row: pixels [0, n) of a row, n = width rounded down to 8; chroma replicated by `sx` (0 or 1). Returns n; the caller does
// the rest with the portable row. r, g, b get the SDR signal in [0, 1]; `scratch` (3 x width floats) and `todo` (width) are the
// caller's work space.
uint32_t RowAvx2(const Chain& chain, const Yuv& k, const uint8_t* y, const uint8_t* u, const uint8_t* v, int sx, uint32_t width,
                 float* r, float* g, float* b, float* scratch, int32_t* todo);
uint32_t RowAvx2(const Chain& chain, const Yuv& k, const uint16_t* y, const uint16_t* u, const uint16_t* v, int sx, uint32_t width,
                 float* r, float* g, float* b, float* scratch, int32_t* todo);

}  // namespace dlssvid::hdrrow
