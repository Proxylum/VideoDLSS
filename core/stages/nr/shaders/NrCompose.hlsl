// Neural Rendering resolve (stage 6): the model's answer -> the edited frame, with masks and the optional
// temporal filter.
//   mode 0 (model at full resolution): edited = model
//   mode 1 (model at a reduced / enlarged working size, HACK "ratio transfer" after OptiScaler_DLSSNR):
//           edited = lerp(original, original * clamp(model_up / proxy_up, 1/maxRatio, maxRatio), transfer)
//           — the model's edit is carried onto the full-resolution frame as a per-channel ratio between what it
//           returned and what it saw (`proxy`), so full-resolution detail survives a reduced model resolution.
//   masks (ТЗ §5): protect = max(ui, ignore) -> back to the original; skin = max(face, skin) -> edit strength x skinBlend
//   temporal (HACK against flicker of patched DLLs): res = lerp(res, prev(p + mv), temporal * gate * w), gate = 1 - |res - prev| / threshold
// Output clamped to [0, 1], alpha 1.

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
    uint gFlags;        // 1 prev, 2 mv, 4 protect mask, 8 skin mask
    uint3 gPad;
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

[numthreads(8, 8, 1)]
void CSMain(uint3 id : SV_DispatchThreadID) {
    if (id.x >= gSize.x || id.y >= gSize.y) return;
    const float2 uv = (float2(id.xy) + 0.5) / float2(gSize);
    const float3 orig = gOriginal[id.xy].rgb;
    float3 edited;
    if (gMode == 0) {
        edited = gModel[id.xy].rgb;
    } else {
        const float3 m = gModel.SampleLevel(gLinear, uv, 0).rgb;
        const float3 p = gProxy.SampleLevel(gLinear, uv, 0).rgb;
        const float3 ratio = clamp(m / max(p, 1e-3), 1.0 / gMaxRatio, gMaxRatio);
        edited = lerp(orig, orig * ratio, gTransfer);
    }
    float w = 1.0;
    if (gFlags & 4) w *= 1.0 - saturate(gProtect.SampleLevel(gLinear, uv, 0));
    if (gFlags & 8) w *= lerp(1.0, gSkinBlend, saturate(gSkin.SampleLevel(gLinear, uv, 0)));
    float3 res = lerp(orig, edited, w);
    if ((gFlags & 1) && gTemporal > 0.0) {
        float2 off = 0;
        if (gFlags & 2) off = gMv.SampleLevel(gLinear, uv, 0).xy * gMvScale;
        const float2 puv = (float2(id.xy) + 0.5 + off) / float2(gSize);
        const bool inside = all(puv >= 0.0) && all(puv <= 1.0);
        const float3 prev = gPrev.SampleLevel(gLinear, puv, 0).rgb;
        const float gate = 1.0 - saturate(length(res - prev) / max(gThreshold, 1e-4));
        res = lerp(res, prev, gTemporal * gate * w * (inside ? 1.0 : 0.0));  // protected pixels are not filtered either
    }
    gOut[id.xy] = float4(saturate(res), 1.0);
}
