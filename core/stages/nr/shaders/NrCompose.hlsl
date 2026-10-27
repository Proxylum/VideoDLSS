// Neural Rendering resolve (stage 6): the model's answer -> the edited frame, with masks and the optional
// temporal filter.
//   mode 0 (model at full resolution): edited = model
//   mode 1 (model at a reduced / enlarged working size, HACK "ratio transfer" after OptiScaler_DLSSNR):
//           edited = lerp(original, original * clamp(model_up / proxy_up, 1/maxRatio, maxRatio), transfer)
//           — the model's edit is carried onto the full-resolution frame as a per-channel ratio between what it
//           returned and what it saw (`proxy`), so full-resolution detail survives a reduced model resolution.
//   masks (ТЗ §5): protect = max(ui, ignore) -> back to the original; skin = max(face, skin) -> edit strength x skinBlend
//   gain (nr.intensity, PR-7, HACK): res = lerp(original, edited, w * gain) - the edit strength after the model, which always gets
//           DLSSNR.Intensity 1 (PR-2 E1: NGX's own intensity is not an edit strength); above 1 it extrapolates
//   temporal (HACK against flicker of patched DLLs): res = lerp(res, prev(p + mv), temporal * gate * w * min(gain, 1)),
//           gate = 1 - |res - prev| / threshold; below 1 the filter fades with the edit, so gain 0 gives the original exactly
//   control mask (PR-8, HACK "frequency split", variant NR_CONTROL_MASK): c = mask_control at the frame size (grey: v, 1, 1); w *= c.r;
//           res = orig + w * gain * (B * e + (G - B) * e_lo), e = edited - orig, e_lo = LowPass_d(e): CSEditBox (d x d means over the
//           pixels that exist, the last cell of a row / column partial with its nominal centre) -> CSEditGauss (3 x 3) -> LowUp (bilinear
//           in fp32); d = max(2, round(min(W, H) / 180)) of the output frame (E10). The filter's weight x max(G, B): no edit, no filter.
//           Without a mask the default variant runs - the resolve of PR-7 bit for bit (same DXIL)
// Order of the strength controls (spec §2b): ratio transfer -> masks (w, the control mask's R included) -> G / B split -> gain, once
// after the last model pass -> temporal filter -> saturate. Output clamped to [0, 1], alpha 1.
// The edited colour is NrEdited.hlsli, a block of statements: the default variant (NR_CONTROL_MASK 0) is token for token the resolve of
// PR-7 (tests/unit/frozen/NrCompose_pr7.hlsl; the unit tests compare the bytecode and the output).

#ifndef NR_CONTROL_MASK
#define NR_CONTROL_MASK 0
#endif

cbuffer Constants : register(b0) {
    uint2 gSize;        // output size
    uint2 gWorkSize;    // model / proxy size
    uint gMode;
    float gTransfer;
    float gMaxRatio;
    float gTemporal;
    float gThreshold;
    float gSkinBlend;
    float2 gMvScale;    // guide px -> output px
    uint gFlags;        // 1 prev, 2 mv, 4 protect mask, 8 skin mask; the mask variant: 16 colour control mask, 32 low band bound (split)
    float gGain;        // nr.intensity: the edit strength after the model (PR-7); 1 = the model's picture
#if NR_CONTROL_MASK
    uint gLowSizePacked;  // the low band grid of the control mask's split: width | height << 16 (ceil(W/d) x ceil(H/d), sides <= 8192)
    uint gLowDiv;         // d
#else
    uint2 gPad;
#endif
};

Texture2D<float4> gOriginal : register(t0);  // full-resolution tone-mapped frame
Texture2D<float4> gModel : register(t1);     // the model's output (work size)
Texture2D<float4> gProxy : register(t2);     // what the model saw (work size; mode 1)
Texture2D<float4> gPrev : register(t3);      // previous resolved frame (full size)
Texture2D<float2> gMv : register(t4);        // mv_dlss (guide size, px, backward)
Texture2D<float> gProtect : register(t5);    // [0,1]
Texture2D<float> gSkin : register(t6);       // [0,1]
RWTexture2D<float4> gOut : register(u0);
SamplerState gLinear : register(s0);
#if NR_CONTROL_MASK
Texture2D<float4> gControl : register(t7);  // mask_control at the frame size: R16_UNORM (grey) or R16G16B16A16_UNORM
Texture2D<float4> gLow : register(t8);      // CSEditGauss: the box means; CSMain: the low band of the edit (RGBA32F grid)

uint2 LowSize() { return uint2(gLowSizePacked & 0xFFFF, gLowSizePacked >> 16); }
// The low band at pixel p: bilinear in fp32 through Load, cell i centred at (i + 0.5) d, edges clamped (the CPU twin: NrLowpassUpCpu).
float3 LowUp(uint2 p) {
    const float2 s = (float2(p) + 0.5) / float(gLowDiv) - 0.5;
    const float2 f0 = floor(s), f = s - f0;
    const int2 hi = int2(LowSize()) - 1;
    const int2 a = clamp(int2(f0), 0, hi), b = clamp(int2(f0) + 1, 0, hi);
    const float3 t = lerp(gLow[int2(a.x, a.y)].rgb, gLow[int2(b.x, a.y)].rgb, f.x);
    const float3 u = lerp(gLow[int2(a.x, b.y)].rgb, gLow[int2(b.x, b.y)].rgb, f.x);
    return lerp(t, u, f.y);
}
#endif

[numthreads(8, 8, 1)]
void CSMain(uint3 id : SV_DispatchThreadID) {
    if (id.x >= gSize.x || id.y >= gSize.y) return;
    const float2 uv = (float2(id.xy) + 0.5) / float2(gSize);
    const float3 orig = gOriginal[id.xy].rgb;
#include "NrEdited.hlsli"
    float w = 1.0;
    if (gFlags & 4) w *= 1.0 - saturate(gProtect.SampleLevel(gLinear, uv, 0));
    if (gFlags & 8) w *= lerp(1.0, gSkinBlend, saturate(gSkin.SampleLevel(gLinear, uv, 0)));
#if NR_CONTROL_MASK
    const float4 cm = gControl[id.xy];
    const float3 c = (gFlags & 16) ? cm.rgb : float3(cm.r, 1.0, 1.0);  // grey: (v, 1, 1) - the edit x v
    w *= c.r;                                                          // white: w * 1 is w exactly
    const float3 e = edited - orig;
    float3 mixed = c.b * e;
    if (gFlags & 32) mixed += (c.g - c.b) * LowUp(id.xy);  // HACK: the frequency split of the edit (PR-8); G = B adds nothing
    float3 res = orig + (w * gGain) * mixed;
#else
    float3 res = lerp(orig, edited, w * gGain);  // gain 1: w * 1 is w exactly - the resolve before PR-7
#endif
    if ((gFlags & 1) && gTemporal > 0.0) {
        float2 off = 0;
        if (gFlags & 2) off = gMv.SampleLevel(gLinear, uv, 0).xy * gMvScale;
        const float2 puv = (float2(id.xy) + 0.5 + off) / float2(gSize);
        const bool inside = all(puv >= 0.0) && all(puv <= 1.0);
        const float3 prev = gPrev.SampleLevel(gLinear, puv, 0).rgb;
        const float gate = 1.0 - saturate(length(res - prev) / max(gThreshold, 1e-4));
        float k = gTemporal * gate * w;
#if NR_CONTROL_MASK
        k *= max(c.g, c.b);  // no edit (R = 0, or G = B = 0): no filter either
#endif
        if (gGain < 1.0) k *= gGain;  // below 1 the filter fades with the edit (gain 0: the original); at 1 and above as before
        res = lerp(res, prev, k * (inside ? 1.0 : 0.0));  // protected pixels are not filtered either
    }
    gOut[id.xy] = float4(saturate(res), 1.0);
}

#if NR_CONTROL_MASK
// The low band of the edit, step 1: one thread per d x d cell - the mean of edited - orig over the pixels of the cell that exist (an
// edge cell has fewer), summed row by row. A non-finite edit counts as 0: a NaN of the model stays one pixel instead of a 3d x 3d
// block. Writes u0 (the box means, RGBA32F).
[numthreads(8, 8, 1)]
void CSEditBox(uint3 cell : SV_DispatchThreadID) {
    const uint2 ls = LowSize();
    if (cell.x >= ls.x || cell.y >= ls.y) return;
    const uint2 p0 = cell.xy * gLowDiv, p1 = min(p0 + gLowDiv, gSize);
    float3 sum = 0;
    for (uint y = p0.y; y < p1.y; ++y) {
        float3 row = 0;
        for (uint x = p0.x; x < p1.x; ++x) {
            const uint3 id = uint3(x, y, 0);
            const float2 uv = (float2(id.xy) + 0.5) / float2(gSize);  // the resolve's sample points
            const float3 orig = gOriginal[id.xy].rgb;
#include "NrEdited.hlsli"
            const float3 e = edited - orig;
            row += select(isfinite(e), e, 0.0);
        }
        sum += row;
    }
    const uint2 n = p1 - p0;
    gOut[cell.xy] = float4(sum / float(n.x * n.y), 1.0);
}

// Step 2: Gauss [1 2 1] x [1 2 1] / 16 on the grid, edges clamped; reads t8 (the box means), writes u0 (the low band).
[numthreads(8, 8, 1)]
void CSEditGauss(uint3 cell : SV_DispatchThreadID) {
    const int2 hi = int2(LowSize()) - 1;
    if (cell.x > uint(hi.x) || cell.y > uint(hi.y)) return;
    const int2 p = int2(cell.xy);
#define BOX(dx, dy) gLow[clamp(p + int2(dx, dy), int2(0, 0), hi)].rgb
    const float3 r0 = BOX(-1, -1) + 2 * BOX(0, -1) + BOX(1, -1);
    const float3 r1 = BOX(-1, 0) + 2 * BOX(0, 0) + BOX(1, 0);
    const float3 r2 = BOX(-1, 1) + 2 * BOX(0, 1) + BOX(1, 1);
#undef BOX
    gOut[cell.xy] = float4((r0 + 2 * r1 + r2) / 16.0, 1.0);
}
#endif
