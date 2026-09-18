// MV arrows on a grid (ТЗ §6): instanced line list, the vertex shader reads the MV texture
// directly so nothing is copied to the CPU. 3 segments (shaft + two head strokes) per cell.

cbuffer ArrowConstants : register(b0) {
    float4 gCell;    // cell rect in render-target pixels (x, y, w, h)
    float4 gView;    // offsetX, offsetY, zoom, 0
    float4 gImage;   // image w, h, arrow step (px), vector scale (screen px per image px of motion)
    float4 gTarget;  // render-target w, h, grid columns, grid rows
    float4 gTexSize; // mv texture w, h, 0, 0
    float4 gColor;
};

Texture2D<float4> gMv : register(t0);

struct VSOut {
    float4 pos : SV_Position;
    float4 color : COLOR0;
};

float4 ToClip(float2 screen) {
    float2 ndc = float2(screen.x / gTarget.x * 2 - 1, 1 - screen.y / gTarget.y * 2);
    return float4(ndc, 0, 1);
}

VSOut VSMain(uint vid : SV_VertexID, uint iid : SV_InstanceID) {
    uint cols = (uint)gTarget.z;
    uint gx = iid % cols, gy = iid / cols;
    float step = gImage.z;
    float2 center = (float2(gx, gy) + 0.5) * step;  // image px
    float2 uv = center / gImage.xy;
    int2 tp = int2(uv * gTexSize.xy);
    float2 v = gMv.Load(int3(tp, 0)).rg;
    float2 tip = center + v * gImage.w;
    float len = length(tip - center);
    float2 dir = len > 1e-3 ? (tip - center) / len : float2(1, 0);
    float2 nrm = float2(-dir.y, dir.x);
    float head = min(len * 0.35, step * 0.3);
    float2 p;
    // segment 0: center->tip, 1: tip->tip-head*(dir+nrm*0.6), 2: tip->tip-head*(dir-nrm*0.6)
    uint seg = vid / 2, end = vid & 1;
    if (seg == 0) p = end ? tip : center;
    else if (seg == 1) p = end ? tip - head * (dir + nrm * 0.6) : tip;
    else p = end ? tip - head * (dir - nrm * 0.6) : tip;
    float2 screen = gView.xy + p * gView.z;
    VSOut o;
    o.pos = ToClip(screen);
    o.color = float4(gColor.rgb, len > 0.25 ? 1.0 : 0.25);
    return o;
}

float4 PSMain(VSOut i) : SV_Target { return float4(i.color.rgb, 1) * i.color.a + float4(0, 0, 0, 0); }
