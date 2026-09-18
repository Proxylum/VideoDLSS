# Архитектура

Полная целевая архитектура — ТЗ §4 (граф стадий, кэш кадров, контракты). Здесь — то, что
реализовано, и хаки, которые нужно помнить.

## Состояние после этапов 0–4

```
VideoDecoder ──CpuFrame(YUV420P)──▶ Pipeline ──▶ VideoEncoder
   FFmpeg                              │            FFmpeg + NVENC / ffv1
   (sw | NVDEC→CPU)                    ▼            audio: stream copy
                              [ PassthroughStage ]
                                       │ upload / readback
                                       ▼
                              GpuFrameCache (ring of D3D12 buffers)
                                       │
                              D3D12Device (single device + direct queue)
                                       │ shared heap → external memory
                              CudaInterop (cudaImportExternalMemory)
```

- `IStage` — `Init(StageConfig, D3D12Device)`, `Process(FrameContext)`, `Shutdown()`; `StageConfig.params` — JSON.
- `Pipeline` — линейный список; граф/планировщик появится, когда стадий станет больше одной (этап 2).
- `GpuFrameCache` — кольцо из N слотов; сейчас в слоте один буфер с байтами кадра. Стадии 1–3 добавят
  в слот текстуры пассов (`color` RGBA16F, `depth_raw` R32F, `mv_raw` RG16F …).
- `CudaInterop` — выбор CUDA-девайса по LUID адаптера D3D12; импорт буфера, созданного на
  `D3D12_HEAP_FLAG_SHARED`. Синхронизация пока блокирующая (`cudaDeviceSynchronize` + fence);
  внешние семафоры (`cudaImportExternalSemaphore` ↔ `ID3D12Fence`) — этап 2.

### Слой пассов (этап 1)

```
PassImage (CPU, interleaved, u8/u16/f16/f32)        core/passes/PassImage.h
   │  PassKind: color_source, depth_raw, depth_dlss, mv_raw, mv_dlss, mask, color_sr/nr/fg, result
   ▼
PassFile  ──▶ formats/ExrIO · PngIO · TiffIO · NpzIO · RawIO   (диспетчер по расширению)
   ▼
PassWriter / PassReader  (папка = файлы <pass>_%06d.<ext> + manifest.json)   core/passes/PassSequence.h
   ▼
ExportPass / ExportLayeredExr / presets (nuke, comfyui, rawdlss)
convert/: DepthConvert (raw ↔ reverse-Z), MvConvert (InvertFlow, dilation, scale), ColorConvert (YUV → RGB)
```

- Пассы живут на CPU (`PassImage`) и на диске; GPU-текстуры пассов добавляются в слот
  `GpuFrameCache` вместе с первой считающей стадией (этап 2). Дисковый кэш пассов (ТЗ §4) — это
  те же папки `PassWriter`/`PassReader` в каталоге проекта.
- Все формулы — `docs/conventions.md`; они реализованы один раз в `core/convert/` и не дублируются в стадиях.
- `Manifest` — единственный источник геометрии для бинарных дампов и канонических имён каналов;
  `PassReader::Validate` формулирует несовпадения (разрешение, число кадров, пропуски) одним сообщением.

### Стадия Depth (этап 2)

```
CpuFrame(YUV) ─ColorConvert─▶ RGB F32 ─▶ DepthStage (окно N кадров, overlap) ─▶ IDepthEstimator
                                                                                   ├─ TrtDepthEstimator (da3, vda): resize→NCHW→TensorRT FP16→disparity/depth→guided upsample
                                                                                   ├─ WorkerDepthEstimator (worker:da3|vda|icdepth|stub): python depth_worker, .npz через scratch
                                                                                   └─ StubDepthEstimator (тесты)
                                                            ▼
                             FillInvalidDepth → TemporalStabilizer (scale/shift, окно 8) → TAE → PassWriter(depth_raw, depth_dlss) + GpuFrameCache slot texture R32F
```

- TensorRT грузится в рантайме (`ml/TrtLoader`): C-точки входа заголовков (`createInferRuntime_INTERNAL` …)
  определены у нас и форвардят в `nvinfer_10.dll`; import-библиотеки не нужны, без DLL бэкенды `da3`/`vda`
  выдают понятную ошибку, остальное работает.
- ONNX под конкретную геометрию (`<model>_<T>x<H>x<W>.onnx`) экспортируется при первом обращении скриптом
  из venv; engine кэшируется по хэшу ONNX + GPU + версия TensorRT + fp16.
- `IStage::Finish()` добавлен для оконных стадий (сброс хвоста окна после последнего кадра).
- Метрика TAE без MV — прокси (|d_t − d_{t−1}| / mean d); warp-версия — этап 3.

### Стадия Motion vectors (этап 3)

```
NVDEC (CUDA NV12, primary context) ──GpuFrame──▶ FlowStage ──▶ IFlowEstimator
   │ (CPU-копия только по запросу)                 │  задержка в 1 кадр    ├─ OfaFlowEstimator: cudaMemcpy2D NV12 → буферы nvofapi → S10.5 → float
   └──CudaInterop::CopyNv12──▶ GpuFrameCache slot  │                      ├─ TrtFlowEstimator (searaft): resize → 2×NCHW → TensorRT → ScaleMv
      (shared heap, NV12 layout, без CPU)          ▼                      └─ StubFlowEstimator
                                     mv_raw[t] (forward) ─ForwardFlowToBackwardMv(+depth)─▶ mv_dlss[t+1] ─▶ PassWriter + GpuFrameCache textures
                                                                            └─ WarpPsnr(prev, cur, mv_dlss) → статистика/манифест
```

- `FrameContext.gpu` — NVDEC-кадр на устройстве; `Pipeline::ProcessFrame(frame, gpu)`; `RunPipeline`/`RunFlow`
  передают его при `--hwaccel cuda`. `GpuFrameCache::AttachCuda` создаёт слоты на shared-heap и импортирует их в
  CUDA, `UploadFromGpuFrame` копирует NV12 device-to-device (readback распаковывает в YUV420P).
- OFA держит копию предыдущего NV12-кадра в собственной device-памяти (декодер переиспользует свой буфер).
- Глубина для окклюзий в `mv_dlss` берётся из папки `depth_raw` (`--depth-dir`) или из слота кэша, если DepthStage
  отработала в том же пайплайне.


### Вьюпорт (этап 4)

```
FrameStore (GPU-кэш кадров)                    ViewportRenderer (один composite-шейдер)
  ├─ loader-потоки (2–4): пассы через PassReader,   ├─ Composite.hlsl: до 5 слоёв, colormaps, MV HSV/magnitude,
  │  видео через VideoDecoder (один декодер на       │  маски fill/contour, blend normal/difference/multiply/screen,
  │  источник, seek к ключевому кадру + проход        │  wipe, checkerboard вне кадра; 2x2 = 4 draw с viewport/scissor
  │  вперёд, кадры «по пути» из окна префетча        ├─ Arrows.hlsl: инстансированные линии, VS читает MV-текстуру
  │  сохраняются) → RGBA16F / R32F / RG32F / R8      ├─ swapchain (QWindow) или offscreen RGBA8 (PNG, CLI, тесты)
  ├─ Update() на render-потоке: upload ≤ N/кадр,      └─ ReadTexel: 1×1 readback для пробника
  │  LRU-вытеснение по бюджету VRAM вне окна ±prefetch
  └─ статусы Missing / Queued / Ready
ViewportState (JSON в проекте) ── Project (*.dlssvid.json: source, passes, result, stages, viewport)
AppModel (Qt) ── ViewportWindow / панели ── TaskQueue (dlssvid <stage> как процесс, прогресс из "N/M frames")
```

- Один и тот же путь для GUI и CLI: `dlssvid render` собирает `ViewportState` из проекта и опций, ждёт
  `FrameStore::WaitForCurrent()` и рендерит offscreen; GUI рендерит в swapchain по таймеру 16 мс и
  подхватывает кадры по мере готовности (`FrameStore::Update()`).
- Видео в кэше — CPU-декод (FFmpeg, frame/slice-потоки) → `Yuv420pToRgba16f` (F16C) → upload. На 1080p
  (RTX 4070 Ti SUPER, Ryzen 7 7700X): холодный скраббинг одного источника 12 мс/кадр (82 fps), сетка 2x2
  из четырёх 1080p-источников 27 мс/кадр (37 fps), тёплый (кадры в кэше) < 1 мс, случайные прыжки по
  H.264 с GOP 250 — 40 мс (seek + декод от ключевого кадра); composite сам по себе 0.2–2.5 мс.
- Проект хранит относительные пути (переносим вместе с папкой), состояние вьюпорта и конфиги стадий;
  панель проекта запускает `dlssvid depth|flow` с параметрами из JSON стадии.

## Хаки и временные решения

| Где | Что | Почему | Когда убираем |
|---|---|---|---|
| `VideoDecoder::ReceiveFrame` | NVDEC-кадры скачиваются на CPU (`av_hwframe_transfer_data`) | этап 0 проверяет bit-exact путь, а не производительность | этап 2: NVDEC → CUDA → D3D12 без CPU |
| `VideoDecoder::ConvertFrame` | не-8-бит-4:2:0 источники → swscale → yuv420p (lossy) | стадии работают в 8-бит 4:2:0 до появления GPU-конверсии | этап 2: RGBA16F на GPU |
| `PassthroughStage` | round-trip upload → readback каждого кадра | доказательство корректности пути GPU | остаётся как диагностический режим |
| `MvConvert::InvertFlow` | инверсия flow сплэттингом с округлением до пикселя, дыры — BFS-заполнением | простая детерминированная инверсия без субпиксельного ресемплинга | этап 3: сравнить с backward-warp SEA-RAFT/OFA, при необходимости заменить |
| `ColorConvert` | хрома 4:2:0 реплицируется (nearest), без интерполяции | детерминизм и обратимость поблочно | этап 3+: конверсия YUV → RGB на GPU с настраиваемым фильтром |
| `DepthStage`, `TrtFlowEstimator` | препроцессинг/апсемпл/guided filter на CPU; входы TensorRT копируются host→device | модели и контракты проверяются на CPU-пути; шейдерной инфраструктуры ещё нет | этап 4–5: препроцессинг compute-шейдером на общем D3D12-девайсе, вход TensorRT из CUDA-буфера |
| `FlowStage::Process` | NV12-кадр копируется в собственный device-буфер стадии | декодер отдаёт один и тот же буфер на следующем кадре | этап 4+: кольцо NVDEC-кадров в `GpuFrameCache` |
| `TemporalAlignmentError` | статичный прокси, если `--mv-dir` не задан | обратная совместимость | — (warp-версия есть) |
| `FrameStore::LoadVideo` | видео для вьюпорта декодируется на CPU и конвертируется в RGBA16F (`Yuv420pToRgba16f`, F16C) перед upload | NVDEC-кольцо и YUV→RGB на GPU появятся вместе с GPU-препроцессингом этапа 5; CPU-путь укладывается в ≥ 30 fps на 1080p | этап 5: NVDEC → CUDA → D3D12-текстура, конверсия compute-шейдером |
| `ViewportRenderer::RenderInto` | каждый кадр — `ExecuteAndWait` (синхронный submit) | простота; composite 0.2–2.5 мс, узкое место — загрузка кадров | этап 5+: fence-ринг и несколько кадров в полёте |
| `ViewportWindow` | подписи ячеек и пробник — виджеты Qt рядом с вьюпортом, в PNG-скриншот их вписывает `QPainter` | Qt не рисует поверх дочернего нативного окна со swapchain | — (по плану этапа 4) |

Хаки в коде помечаются `// HACK:` (ТЗ §8); эмуляция джиттера для DLSS SR и скрытый swapchain FG
появятся в этапах 5 и 7 и будут описаны здесь.

## Ошибки и деградация
- Отсутствие NVIDIA GPU: D3D12 падает на WARP, CUDA interop недоступен, NVENC-кодеки не найдены —
  всё сообщается в логе и `dlssvid info`; `--codec ffv1` и `--warp` работают везде.
- Ошибки FFmpeg/D3D12/CUDA — исключение `dlssvid::Error` с кодом и текстом; CLI печатает и возвращает 1.
