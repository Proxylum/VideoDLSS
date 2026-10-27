// The edited colour before the masks: mode 0 - the model's own answer, mode 1 - its edit carried over as a ratio (NrCompose.hlsl).
// PR-8: one text for the resolve and for the low band of the control mask. A block of statements, not a function: the default variant
// of NrCompose stays token for token the resolve of PR-7 (tests/unit/frozen). Needs id, uv and orig in scope; declares edited.
    float3 edited;
    if (gMode == 0) {
        edited = gModel[id.xy].rgb;
    } else {
        const float3 m = gModel.SampleLevel(gLinear, uv, 0).rgb;
        const float3 p = gProxy.SampleLevel(gLinear, uv, 0).rgb;
        const float3 ratio = clamp(m / max(p, 1e-3), 1.0 / gMaxRatio, gMaxRatio);
        edited = lerp(orig, orig * ratio, gTransfer);
    }
