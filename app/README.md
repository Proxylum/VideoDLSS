# app (stage 4)

Qt 6.8 Widgets shell: project panel, viewport (single / overlay / 2x2 grid), timeline, inspector, log.
`app/viewport/` holds the HLSL shaders for colour maps, overlay blending and wipe. The viewport renders
with the same D3D12 device as the pipeline. Nothing here until stage 4.
