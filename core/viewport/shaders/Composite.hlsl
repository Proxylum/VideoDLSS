// dlssvid viewport composite (ТЗ §6): one pixel shader for colour maps, overlays, blend modes and wipe.
// Shared by the single, overlay and 2x2 modes (a 2x2 view is four draws with different cell rects).

#define MAX_LAYERS 5

// LayerParams::flags
#define LF_VISIBLE 1u
#define LF_SOLO 2u
#define LF_INVERT 4u
// LayerParams::kind
#define KIND_NONE 0u
#define KIND_COLOR 1u   // RGBA16F, display-referred [0,1]
#define KIND_SCALAR 2u  // R32F
#define KIND_MV 3u      // RG32F pixels
#define KIND_MASK 4u    // R8_UNORM
// display modes (ViewportState::DisplayMode)
#define DM_COLOR 0u
#define DM_GRAYSCALE 1u
#define DM_VIRIDIS 2u
#define DM_TURBO 3u
#define DM_MV_HSV 4u
#define DM_MV_ARROWS 5u
#define DM_MV_MAGNITUDE 6u
#define DM_MASK_FILL 7u
#define DM_MASK_CONTOUR 8u
// blend modes
#define BM_NORMAL 0u
#define BM_DIFFERENCE 1u
#define BM_MULTIPLY 2u
#define BM_SCREEN 3u

struct LayerParams {
    uint display;
    uint blend;
    uint flags;
    uint kind;
    float minValue;
    float maxValue;
    float opacity;
    float mvScale;   // magnitude that maps to full saturation / white
    float texW;      // layer texture size (may differ from the image size, e.g. mv_dlss at target res)
    float texH;
    float pad0;
    float pad1;
};

cbuffer DrawConstants : register(b0) {
    float4 gCell;        // x, y, w, h of the cell in render-target pixels
    float4 gView;        // offsetX, offsetY (screen px of image origin, relative to the target), zoom, linearFilter (0/1)
    float4 gImage;       // image width, height (reference size in image px), checker size, 0
    uint gLayerCount;
    uint gAnySolo;
    uint gWipeEnabled;
    uint gWipeVertical;
    float gWipePos;
    uint gWipeA;
    uint gWipeB;
    uint gCellState;     // 0 ready, 1 loading, 2 no frame (stage 9: a plate instead of an empty cell)
    float4 gBackground;  // outside-image colour
    LayerParams gLayers[MAX_LAYERS];
};

Texture2D<float4> gTex0 : register(t0);
Texture2D<float4> gTex1 : register(t1);
Texture2D<float4> gTex2 : register(t2);
Texture2D<float4> gTex3 : register(t3);
Texture2D<float4> gTex4 : register(t4);
SamplerState gPoint : register(s0);
SamplerState gLinear : register(s1);

struct VSOut {
    float4 pos : SV_Position;
};

// Fullscreen triangle; the pixel shader clips to the cell rect itself via scissor set by the caller.
VSOut VSMain(uint id : SV_VertexID) {
    VSOut o;
    float2 uv = float2((id << 1) & 2, id & 2);
    o.pos = float4(uv * float2(2, -2) + float2(-1, 1), 0, 1);
    return o;
}

float4 SampleLayer(uint i, float2 uv) {
    // uv in [0,1] over the image; layers with another resolution are stretched to the image.
    [branch] if (gView.w > 0.5) {
        switch (i) {
            case 0: return gTex0.SampleLevel(gLinear, uv, 0);
            case 1: return gTex1.SampleLevel(gLinear, uv, 0);
            case 2: return gTex2.SampleLevel(gLinear, uv, 0);
            case 3: return gTex3.SampleLevel(gLinear, uv, 0);
            default: return gTex4.SampleLevel(gLinear, uv, 0);
        }
    }
    switch (i) {
        case 0: return gTex0.SampleLevel(gPoint, uv, 0);
        case 1: return gTex1.SampleLevel(gPoint, uv, 0);
        case 2: return gTex2.SampleLevel(gPoint, uv, 0);
        case 3: return gTex3.SampleLevel(gPoint, uv, 0);
        default: return gTex4.SampleLevel(gPoint, uv, 0);
    }
}

float4 LoadLayer(uint i, int2 p) {
    switch (i) {
        case 0: return gTex0.Load(int3(p, 0));
        case 1: return gTex1.Load(int3(p, 0));
        case 2: return gTex2.Load(int3(p, 0));
        case 3: return gTex3.Load(int3(p, 0));
        default: return gTex4.Load(int3(p, 0));
    }
}

// ---- colour maps (polynomial fits of matplotlib viridis and Google turbo, x in [0,1]) ----
float3 Viridis(float t) {
    const float3 c0 = float3(0.2777273272234177, 0.005407344544966578, 0.3340998053353061);
    const float3 c1 = float3(0.1050930431085774, 1.404613529898575, 1.384590162594685);
    const float3 c2 = float3(-0.3308618287255563, 0.214847559468213, 0.09509516302823659);
    const float3 c3 = float3(-4.634230498983486, -5.799100973351585, -19.33244095627987);
    const float3 c4 = float3(6.228269936347081, 14.17993336680509, 56.69055260068105);
    const float3 c5 = float3(4.776384997670288, -13.74514537774601, -65.35303263337234);
    const float3 c6 = float3(-5.435455855934631, 4.645852612178535, 26.3124352495832);
    return saturate(c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6))))));
}

float3 Turbo(float t) {
    const float4 kR4 = float4(0.13572138, 4.61539260, -42.66032258, 132.13108234);
    const float4 kG4 = float4(0.09140261, 2.19418839, 4.84296658, -14.18503333);
    const float4 kB4 = float4(0.10667330, 12.64194608, -60.58204836, 110.36276771);
    const float2 kR2 = float2(-152.94239396, 59.28637943);
    const float2 kG2 = float2(4.27729857, 2.82956604);
    const float2 kB2 = float2(-89.90310912, 27.34824973);
    float4 v4 = float4(1.0, t, t * t, t * t * t);
    float2 v2 = v4.zw * v4.z;
    return saturate(float3(dot(v4, kR4) + dot(v2, kR2), dot(v4, kG4) + dot(v2, kG2), dot(v4, kB4) + dot(v2, kB2)));
}

float3 Hsv2Rgb(float3 c) {
    float3 p = abs(frac(c.xxx + float3(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0);
    return c.z * lerp(1.0, saturate(p - 1.0), c.y);
}

// Returns the shaded colour of layer i at image uv and its coverage (0 = transparent).
float4 Shade(uint i, float2 uv, float2 imagePx) {
    LayerParams L = gLayers[i];
    if (L.kind == KIND_NONE) return float4(0, 0, 0, 0);
    float4 s = SampleLayer(i, uv);
    float3 rgb = 0;
    float coverage = 1;
    uint mode = L.display;
    if (mode == DM_COLOR) {
        rgb = (L.kind == KIND_COLOR) ? s.rgb : s.rrr;
        if (L.flags & LF_INVERT) rgb = 1 - rgb;
    } else if (mode == DM_GRAYSCALE || mode == DM_VIRIDIS || mode == DM_TURBO) {
        float v = s.r;
        if (L.kind == KIND_COLOR) v = dot(s.rgb, float3(0.2126, 0.7152, 0.0722));
        if (L.kind == KIND_MV) v = length(s.rg);
        float range = max(L.maxValue - L.minValue, 1e-12);
        float t = saturate((v - L.minValue) / range);
        if (L.flags & LF_INVERT) t = 1 - t;
        if (mode == DM_GRAYSCALE) rgb = t.xxx;
        else if (mode == DM_VIRIDIS) rgb = Viridis(t);
        else rgb = Turbo(t);
    } else if (mode == DM_MV_HSV || mode == DM_MV_ARROWS) {
        float2 v = s.rg;
        float mag = length(v);
        float scale = max(L.mvScale, 1e-6);
        float hue = (atan2(-v.y, v.x) / 6.28318530718) + 0.5;  // right = 0.5 (cyan-ish), standard flow wheel
        float sat = saturate(mag / scale);
        if (L.flags & LF_INVERT) sat = 1 - sat;
        rgb = (mode == DM_MV_ARROWS) ? lerp(0.15, 0.35, sat).xxx : Hsv2Rgb(float3(frac(hue), sat, 1.0));
    } else if (mode == DM_MV_MAGNITUDE) {
        float t = saturate(length(s.rg) / max(L.mvScale, 1e-6));
        if (L.flags & LF_INVERT) t = 1 - t;
        rgb = t.xxx;
    } else if (mode == DM_MASK_FILL) {
        float a = (L.kind == KIND_MASK) ? s.r : saturate(s.r);
        if (L.flags & LF_INVERT) a = 1 - a;
        rgb = float3(1.0, 0.25, 0.1);
        coverage = a;
    } else if (mode == DM_MASK_CONTOUR) {
        // edge where the thresholded mask changes between neighbouring texels
        int2 p = int2(uv * float2(L.texW, L.texH));
        float c = LoadLayer(i, p).r > 0.5 ? 1 : 0;
        float e = 0;
        e += abs(c - (LoadLayer(i, p + int2(1, 0)).r > 0.5 ? 1 : 0));
        e += abs(c - (LoadLayer(i, p + int2(-1, 0)).r > 0.5 ? 1 : 0));
        e += abs(c - (LoadLayer(i, p + int2(0, 1)).r > 0.5 ? 1 : 0));
        e += abs(c - (LoadLayer(i, p + int2(0, -1)).r > 0.5 ? 1 : 0));
        coverage = e > 0 ? 1 : 0;
        rgb = float3(1.0, 0.9, 0.1);
    }
    return float4(rgb, coverage);
}

float3 Blend(uint mode, float3 dst, float3 src, float a) {
    float3 b = src;
    if (mode == BM_DIFFERENCE) b = abs(dst - src);
    else if (mode == BM_MULTIPLY) b = dst * src;
    else if (mode == BM_SCREEN) b = 1 - (1 - dst) * (1 - src);
    return lerp(dst, b, a);
}

float4 PSMain(VSOut i) : SV_Target {
    float2 screen = i.pos.xy;  // render-target pixels
    float2 img = (screen - gView.xy) / gView.z;  // image pixel coordinates
    float2 uv = img / gImage.xy;
    if (any(img < 0) || any(img >= gImage.xy)) {
        // checkerboard outside the image
        float2 cs = floor(screen / max(gImage.z, 1.0));
        float k = fmod(cs.x + cs.y, 2.0);
        return float4(gBackground.rgb * (0.85 + 0.15 * k), 1);
    }
    if (gCellState == 2u) {  // no frame of the base layer at this time: diagonal hatch
        float k = fmod(floor((screen.x + screen.y) / 12.0), 2.0);
        return float4(lerp(float3(0.11, 0.11, 0.12), float3(0.17, 0.17, 0.19), k), 1);
    }
    if (gCellState == 1u) return float4(0.20, 0.21, 0.24, 1);  // still loading: a plain plate

    float3 result = 0;
    bool first = true;
    uint count = min(gLayerCount, MAX_LAYERS);

    // Wipe: the two compared layers are shown side by side, others composite on top.
    if (gWipeEnabled != 0 && gWipeA < count && gWipeB < count) {
        float coord = gWipeVertical != 0 ? uv.x : uv.y;
        uint pick = coord < gWipePos ? gWipeA : gWipeB;
        float4 c = Shade(pick, uv, img);
        result = c.rgb * c.a;
        first = false;
        // thin split line
        float px = gWipeVertical != 0 ? gImage.x : gImage.y;
        if (abs(coord - gWipePos) * px * gView.z < 1.0) return float4(1, 1, 1, 1);
    }

    for (uint l = 0; l < count; ++l) {
        LayerParams L = gLayers[l];
        if ((L.flags & LF_VISIBLE) == 0) continue;
        if (gAnySolo != 0 && (L.flags & LF_SOLO) == 0) continue;
        if (gWipeEnabled != 0 && (l == gWipeA || l == gWipeB)) continue;
        if (L.kind == KIND_NONE) continue;
        float4 c = Shade(l, uv, img);
        float a = L.opacity * c.a;
        if (first) {
            result = c.rgb * c.a * (first && l == 0 ? 1.0 : L.opacity) + result * (1 - c.a);
            first = false;
        } else {
            result = Blend(L.blend, result, c.rgb, a);
        }
    }
    return float4(saturate(result), 1);
}
