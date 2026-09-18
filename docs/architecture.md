# Архитектура

Полная целевая архитектура — ТЗ §4 (граф стадий, кэш кадров, контракты). Здесь — то, что
реализовано, и хаки, которые нужно помнить.

## Состояние после этапов 0–2

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

## Хаки и временные решения

| Где | Что | Почему | Когда убираем |
|---|---|---|---|
| `VideoDecoder::ReceiveFrame` | NVDEC-кадры скачиваются на CPU (`av_hwframe_transfer_data`) | этап 0 проверяет bit-exact путь, а не производительность | этап 2: NVDEC → CUDA → D3D12 без CPU |
| `VideoDecoder::ConvertFrame` | не-8-бит-4:2:0 источники → swscale → yuv420p (lossy) | стадии работают в 8-бит 4:2:0 до появления GPU-конверсии | этап 2: RGBA16F на GPU |
| `PassthroughStage` | round-trip upload → readback каждого кадра | доказательство корректности пути GPU | остаётся как диагностический режим |
| `MvConvert::InvertFlow` | инверсия flow сплэттингом с округлением до пикселя, дыры — BFS-заполнением | простая детерминированная инверсия без субпиксельного ресемплинга | этап 3: сравнить с backward-warp SEA-RAFT/OFA, при необходимости заменить |
| `ColorConvert` | хрома 4:2:0 реплицируется (nearest), без интерполяции | детерминизм и обратимость поблочно | этап 3+: конверсия YUV → RGB на GPU с настраиваемым фильтром |
| `DepthStage` | препроцессинг/апсемпл/guided filter на CPU; кадры для TensorRT копируются host→device | этап 2 проверяет модели и контракты; GPU-путь придёт с CUDA-кадрами | этап 3: NVDEC → CUDA → TensorRT без CPU, апсемпл шейдером |
| `TemporalAlignmentError` | без warp по MV (статичный прокси) | MV нет до этапа 3 | этап 3 |

Хаки в коде помечаются `// HACK:` (ТЗ §8); эмуляция джиттера для DLSS SR и скрытый swapchain FG
появятся в этапах 5 и 7 и будут описаны здесь.

## Ошибки и деградация
- Отсутствие NVIDIA GPU: D3D12 падает на WARP, CUDA interop недоступен, NVENC-кодеки не найдены —
  всё сообщается в логе и `dlssvid info`; `--codec ffv1` и `--warp` работают везде.
- Ошибки FFmpeg/D3D12/CUDA — исключение `dlssvid::Error` с кодом и текстом; CLI печатает и возвращает 1.
