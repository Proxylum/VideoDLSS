# План этапа 3 — motion vectors

Дата: 2026-09-18. Ветка: `stage/03-motion-vectors`. Требования: ТЗ §4 (стадия Motion vectors), §5 (`mv_raw`,
`mv_dlss`), §8 (этап 3: OFA-бэкенд, SEA-RAFT-бэкенд, конвертация в `mv_dlss`, warp-тест PSNR ≥ 28 дБ), §9.
Перенесено из этапа 2: NVDEC → CUDA → D3D12 без копий через CPU; TAE с warp по MV.

## Разведка (2026-09-18)
| Что | Статус | Решение |
|---|---|---|
| NVIDIA Optical Flow | `nvofapi64.dll` из драйвера 591.86 отдаёт `NvOFGetMaxSupportedApiVersion = 0x50` (API 5.0, как в ТЗ); публичные заголовки на GitHub — API 2.0 (2020), SDK 5.x с новыми заголовками/сэмплами — за логином developer.nvidia.com | Работаем с заголовками 2.0 (DLL обратно совместима: `NvOFAPICreateInstanceCuda(NV_OF_API_VERSION)`); grid size 1 запрашиваем по `nvOFGetCaps`. SDK 5.x — в TASK-0011 как «желательно» |
| SEA-RAFT | код BSD-3 (princeton-vl), веса — Google Drive и зеркало HF `Yarimasune/SEA-RAFT` (Spring-M/S, KITTI, Tartan); `bilinear_sampler` = `grid_sample` (ONNX opset 16, TensorRT 10 поддерживает GridSample); encoder — torchvision ResNet-34 | Бэкенд `searaft` через TensorRT из ONNX `Tartan-C-T-TSKH-spring540x960-M.pth` (Spring-M, 4 итерации); лицензия весов не указана автором — в реестре помечено, для коммерческого релиза уточнить |
| FFmpeg CUDA hwcontext | `AV_CUDA_USE_PRIMARY_CONTEXT` → CUDA-кадры NV12 в primary-контексте, совместимы с cudart и OFA | декодер отдаёт `GpuFrame` (CUdeviceptr Y/UV + pitch) без `av_hwframe_transfer_data` |

## Что делаем
1. **`core/io/VideoDecoder`**: режим GPU-кадров — при `hwaccel cuda` кадр остаётся на устройстве (`GpuFrame`:
   NV12 CUdeviceptr + pitch, primary context); CPU-копия делается только если запрошена. `FrameContext.gpu`.
2. **`core/gpu/CudaInterop`**: копия NV12/буфера CUDA → импортированный D3D12-буфер (device-to-device), тест
   bit-exact NVDEC-кадр vs software decode через D3D12 readback — закрывает перенос из этапа 2.
3. **`core/stages/flow/`**
   - `IFlowEstimator`: `Estimate(FlowInput a, FlowInput b, PassImage& flow)` — forward flow a→b в пикселях
     исходного разрешения (`mv_raw`, каналы `u`,`v`, F32); `FlowInput` = CPU RGB и/или `GpuFrame`.
   - `OfaFlowEstimator`: `nvofapi64.dll` в рантайме (`NvOFAPICreateInstanceCuda`), вход NV12 напрямую из
     `GpuFrame` (или ABGR8 из CPU RGB), `NV_OF_PERF_LEVEL_SLOW`, grid 1 (или минимальный поддерживаемый),
     выход `SHORT2` S10.5 → float px, апсемпл сетки до полного разрешения, буфер cost → маска доверия (в `stage_params`).
   - `TrtFlowEstimator` (`searaft`): ONNX `(image1, image2: 1×3×H×W, RGB 0..255) → flow 1×2×H×W`, геометрия —
     исходное разрешение, приведённое к кратному 8 и ограниченное `--max-res` (по умолчанию 720 по короткой
     стороне), векторы масштабируются обратно; экспорт `models/export/export_searaft.py`, реестр `sea-raft-spring-m`.
   - `StubFlowEstimator` (тесты): постоянный сдвиг из конфигурации.
   - `FlowStage : IStage`: пары (t, t+1) с задержкой в один кадр → `mv_raw` в кадре t; `mv_dlss` для кадра t+1 через
     `ForwardFlowToBackwardMv` с глубиной (из папки `depth_raw` или из слота кэша), масштаб под `--target`;
     последний кадр — нулевой flow; запись пассов и GPU-текстур (`mv_raw` RG32F).
4. **`core/convert/Warp`**: `WarpBackward(prev, mv_dlss)` (bilinear), `Psnr`, `WarpPsnr(prev, cur, mv_dlss)`;
   `TemporalAlignmentErrorWarped(prevDepth, curDepth, mv_dlss)` — заменяет статичный прокси этапа 2 в `DepthStage`
   (когда MV доступны).
5. **CLI:** `dlssvid flow -i clip -o passes --backend ofa|searaft|stub [--hwaccel cuda] [--depth-dir] [--target WxH]
   [--max-res 720] [--frames]` — печатает средний warp-PSNR; `dlssvid depth --mv-dir passes/mv_dlss` — TAE с warp.
6. **Тесты:** warp/PSNR на синтетическом сдвиге (точность), стадия со `stub` (пассы, задержка в кадр, нулевой хвост,
   mv_dlss с глубиной, GPU-текстуры), декодер GPU-кадров = CPU-декод (bit-exact, пропуск без NVIDIA), OFA на
   синтетическом сдвиге (пропуск без NVIDIA: PSNR warp ≥ 28 дБ, средний вектор ≈ истинному), TensorRT SEA-RAFT —
   smoke в отчёте, CLI как процесс.

## Приёмка (ТЗ §8–9)
- Пассы `mv_raw`/`mv_dlss` с манифестами бэкендами `ofa` и `searaft`; warp-тест: PSNR ≥ 28 дБ на статичных
  сценах (синтетический сдвиг — в ctest; реальные клипы — TASK-0010).
- NVDEC-кадр доходит до OFA и до D3D12 без копий через CPU; тест bit-exact.
