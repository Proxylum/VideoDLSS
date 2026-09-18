# NVIDIA Image Scaling SDK 1.0.3 (vendored)

Source: https://github.com/NVIDIAGameWorks/NVIDIAImageScaling (MIT, see `LICENSE.txt`).
Files: `NIS_Scaler.h` (HLSL/GLSL kernels NVScaler / NVSharpen), `NIS_Config.h` (C++ constant-buffer
setup and coefficient tables). Used by `core/stages/upscale/NisUpscaler` through
`core/stages/upscale/shaders/Nis.hlsl`, compiled with DXC at build time. Unmodified copies.
