// Tonemap compute shader (stage 6, ТЗ §4 «Tonemap»): colour HDR / linear / SDR -> display-referred sRGB
// [0, 1], what Neural Rendering expects. Input transfer: 0 sRGB (decoded SDR video), 1 linear,
// 2 PQ (SMPTE ST 2084, reference white 203 nits), 3 HLG (BT.2100, 1000-nit peak). Curve: 0 passthrough,
// 1 ACES (Narkowicz fit), 2 Reinhard. Exposure multiplies linear light. sRGB in + passthrough + exposure 1
// is an exact copy (no decode / encode round trip).

cbuffer Constants : register(b0) {
    uint2 gSize;
    uint gCurve;
    uint gTransfer;
    float gExposure;
    float3 gPad;
};

Texture2D<float4> gSrc : register(t0);
RWTexture2D<float4> gDst : register(u0);

float3 SrgbToLinear(float3 c) {
    return select(c <= 0.04045, c / 12.92, pow(max((c + 0.055) / 1.055, 0.0), 2.4));
}

float3 LinearToSrgb(float3 c) {
    c = saturate(c);
    return select(c <= 0.0031308, c * 12.92, 1.055 * pow(c, 1.0 / 2.4) - 0.055);
}

// ST 2084 EOTF: signal -> nits, normalised so that 203 nits (BT.2408 reference white) = 1.0
float3 PqToLinear(float3 n) {
    const float m1 = 0.1593017578125, m2 = 78.84375, c1 = 0.8359375, c2 = 18.8515625, c3 = 18.6875;
    float3 p = pow(max(n, 0.0), 1.0 / m2);
    float3 nits = 10000.0 * pow(max(p - c1, 0.0) / max(c2 - c3 * p, 1e-6), 1.0 / m1);
    return nits / 203.0;
}

// HLG inverse OETF (BT.2100): signal -> scene linear [0, 1]; scaled so that the 1000-nit peak maps 203 nits to 1.0
float3 HlgToLinear(float3 e) {
    const float a = 0.17883277, b = 0.28466892, c = 0.55991073;
    float3 lo = e * e / 3.0;
    float3 hi = (exp((e - c) / a) + b) / 12.0;
    float3 scene = select(e <= 0.5, lo, hi);
    return scene * (1000.0 / 203.0);
}

float3 Aces(float3 x) {  // Narkowicz 2015 fit of the ACES filmic curve
    const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return saturate((x * (a * x + b)) / (x * (c * x + d) + e));
}

float3 Reinhard(float3 x) { return x / (1.0 + x); }

[numthreads(8, 8, 1)]
void CSMain(uint3 id : SV_DispatchThreadID) {
    if (id.x >= gSize.x || id.y >= gSize.y) return;
    float4 src = gSrc[id.xy];
    if (gCurve == 0 && gTransfer == 0 && gExposure == 1.0) {
        gDst[id.xy] = float4(saturate(src.rgb), 1.0);
        return;
    }
    float3 lin;
    if (gTransfer == 0) lin = SrgbToLinear(saturate(src.rgb));
    else if (gTransfer == 2) lin = PqToLinear(saturate(src.rgb));
    else if (gTransfer == 3) lin = HlgToLinear(saturate(src.rgb));
    else lin = max(src.rgb, 0.0);
    lin *= gExposure;
    float3 mapped;
    if (gCurve == 1) mapped = Aces(lin);
    else if (gCurve == 2) mapped = Reinhard(lin);
    else mapped = lin;
    gDst[id.xy] = float4(LinearToSrgb(mapped), 1.0);
}
