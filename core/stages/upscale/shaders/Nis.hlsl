// NVIDIA Image Scaling (NIS 1.0.3, MIT — third_party/nis) compute entry point for stage 5.
// Compiled twice by DXC: NIS_SCALER=1 (NVScaler: scaling + sharpening, in/out ratio 0.5..1)
// and NIS_SCALER=0 (NVSharpen: sharpening only, «artifact reduction only» analogue).
// Bindings follow the SDK's NIS_Main.hlsl: cbuffer = NISConfig, t0 input, t1/t2 coefficients, u0 output.

#define NIS_HLSL 1
#ifndef NIS_SCALER
#define NIS_SCALER 1
#endif
#define NIS_HDR_MODE 0
#define NIS_BLOCK_WIDTH 32
#define NIS_BLOCK_HEIGHT 24
#define NIS_THREAD_GROUP_SIZE 256
#define NIS_USE_HALF_PRECISION 0
#define NIS_VIEWPORT_SUPPORT 0
#define NIS_CLAMP_OUTPUT 1

cbuffer cb : register(b0) {
    float kDetectRatio;
    float kDetectThres;
    float kMinContrastRatio;
    float kRatioNorm;

    float kContrastBoost;
    float kEps;
    float kSharpStartY;
    float kSharpScaleY;

    float kSharpStrengthMin;
    float kSharpStrengthScale;
    float kSharpLimitMin;
    float kSharpLimitScale;

    float kScaleX;
    float kScaleY;

    float kDstNormX;
    float kDstNormY;
    float kSrcNormX;
    float kSrcNormY;

    uint kInputViewportOriginX;
    uint kInputViewportOriginY;
    uint kInputViewportWidth;
    uint kInputViewportHeight;

    uint kOutputViewportOriginX;
    uint kOutputViewportOriginY;
    uint kOutputViewportWidth;
    uint kOutputViewportHeight;

    float reserved0;
    float reserved1;
};

SamplerState samplerLinearClamp : register(s0);
Texture2D in_texture : register(t0);
RWTexture2D<float4> out_texture : register(u0);
#if NIS_SCALER
Texture2D coef_scaler : register(t1);
Texture2D coef_usm : register(t2);
#endif

#include "NIS_Scaler.h"

[numthreads(NIS_THREAD_GROUP_SIZE, 1, 1)]
void main(uint3 blockIdx : SV_GroupID, uint3 threadIdx : SV_GroupThreadID) {
#if NIS_SCALER
    NVScaler(blockIdx.xy, threadIdx.x);
#else
    NVSharpen(blockIdx.xy, threadIdx.x);
#endif
}
