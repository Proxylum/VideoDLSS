// Stub "Neural Rendering" (stage 6): a deterministic local-contrast + tone edit so the NR stage, resolve,
// masks, temporal filter, pass count and model resolution can be tested on WARP without nvngx_dlssnr.dll.
// Not a fallback for the real model: only `--backend stub` selects it.

cbuffer Constants : register(b0) {
    uint2 gSize;
    float gIntensity;
    float gLocalTone;
    float gLocalStructure;
    float3 gPad;
};

Texture2D<float4> gSrc : register(t0);
RWTexture2D<float4> gDst : register(u0);

float3 Fetch(int2 p) {
    p = clamp(p, int2(0, 0), int2(gSize) - 1);
    return gSrc[p].rgb;
}

[numthreads(8, 8, 1)]
void CSMain(uint3 id : SV_DispatchThreadID) {
    if (id.x >= gSize.x || id.y >= gSize.y) return;
    const int2 p = int2(id.xy);
    float3 c = Fetch(p);
    float3 blur = 0;
    [unroll] for (int dy = -1; dy <= 1; ++dy)
        [unroll] for (int dx = -1; dx <= 1; ++dx) blur += Fetch(p + int2(dx, dy));
    blur /= 9.0;
    float3 detail = c - blur;
    float3 edited = c + gIntensity * 0.6 * gLocalStructure * detail;
    float3 toned = smoothstep(0.0, 1.0, edited);
    edited = lerp(edited, toned, saturate(0.25 * gIntensity * gLocalTone));
    gDst[id.xy] = float4(saturate(edited), 1.0);
}
