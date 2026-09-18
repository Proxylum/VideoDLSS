// Resampling compute shader (stage 5): dst pixel (x, y) samples src at ((x + 0.5) * in/out + shift)
// in input pixels. With in == out and a sub-pixel `shift` this is the jitter emulation for DLSS
// (HACK, ТЗ §3); with out > in and shift = 0 it is the naive `bicubic` upscaler used as the A/B
// baseline. Output is clamped to [0, 1] (SDR).

cbuffer Constants : register(b0) {
    float2 gInSize;   // input texture size in px
    float2 gOutSize;  // output size in px
    float2 gShift;    // sampling shift in input px
    uint gFilter;     // 0 = bilinear, 1 = Catmull-Rom
    uint gPad;
};

Texture2D<float4> gSrc : register(t0);
RWTexture2D<float4> gDst : register(u0);
SamplerState gLinear : register(s0);

float4 SampleBilinear(float2 p) { return gSrc.SampleLevel(gLinear, p / gInSize, 0); }

// Catmull-Rom with 9 bilinear fetches (Mahalanobis / Jimenez); exact at texel centres.
float4 SampleCatmullRom(float2 p) {
    float2 texPos1 = floor(p - 0.5) + 0.5;
    float2 f = p - texPos1;
    float2 w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
    float2 w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    float2 w2 = f * (0.5 + f * (2.0 - 1.5 * f));
    float2 w3 = f * f * (-0.5 + 0.5 * f);
    float2 w12 = w1 + w2;
    float2 offset12 = w2 / max(w12, 1e-6);
    float2 texPos0 = (texPos1 - 1.0) / gInSize;
    float2 texPos3 = (texPos1 + 2.0) / gInSize;
    float2 texPos12 = (texPos1 + offset12) / gInSize;
    float4 r = 0;
    r += gSrc.SampleLevel(gLinear, float2(texPos0.x, texPos0.y), 0) * w0.x * w0.y;
    r += gSrc.SampleLevel(gLinear, float2(texPos12.x, texPos0.y), 0) * w12.x * w0.y;
    r += gSrc.SampleLevel(gLinear, float2(texPos3.x, texPos0.y), 0) * w3.x * w0.y;
    r += gSrc.SampleLevel(gLinear, float2(texPos0.x, texPos12.y), 0) * w0.x * w12.y;
    r += gSrc.SampleLevel(gLinear, float2(texPos12.x, texPos12.y), 0) * w12.x * w12.y;
    r += gSrc.SampleLevel(gLinear, float2(texPos3.x, texPos12.y), 0) * w3.x * w12.y;
    r += gSrc.SampleLevel(gLinear, float2(texPos0.x, texPos3.y), 0) * w0.x * w3.y;
    r += gSrc.SampleLevel(gLinear, float2(texPos12.x, texPos3.y), 0) * w12.x * w3.y;
    r += gSrc.SampleLevel(gLinear, float2(texPos3.x, texPos3.y), 0) * w3.x * w3.y;
    return r;
}

[numthreads(8, 8, 1)]
void CSMain(uint3 id : SV_DispatchThreadID) {
    if (id.x >= (uint)gOutSize.x || id.y >= (uint)gOutSize.y) return;
    float2 p = (float2(id.xy) + 0.5) * (gInSize / gOutSize) + gShift;
    float4 c = gFilter == 1 ? SampleCatmullRom(p) : SampleBilinear(p);
    gDst[id.xy] = float4(saturate(c.rgb), 1.0);
}
