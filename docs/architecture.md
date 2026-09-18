# Архитектура

Полная целевая архитектура — ТЗ §4 (граф стадий, кэш кадров, контракты). Здесь — то, что
реализовано, и хаки, которые нужно помнить.

## Состояние после этапа 0

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

## Хаки и временные решения

| Где | Что | Почему | Когда убираем |
|---|---|---|---|
| `VideoDecoder::ReceiveFrame` | NVDEC-кадры скачиваются на CPU (`av_hwframe_transfer_data`) | этап 0 проверяет bit-exact путь, а не производительность | этап 2: NVDEC → CUDA → D3D12 без CPU |
| `VideoDecoder::ConvertFrame` | не-8-бит-4:2:0 источники → swscale → yuv420p (lossy) | стадии работают в 8-бит 4:2:0 до появления GPU-конверсии | этап 2: RGBA16F на GPU |
| `PassthroughStage` | round-trip upload → readback каждого кадра | доказательство корректности пути GPU | остаётся как диагностический режим |

Хаки в коде помечаются `// HACK:` (ТЗ §8); эмуляция джиттера для DLSS SR и скрытый swapchain FG
появятся в этапах 5 и 7 и будут описаны здесь.

## Ошибки и деградация
- Отсутствие NVIDIA GPU: D3D12 падает на WARP, CUDA interop недоступен, NVENC-кодеки не найдены —
  всё сообщается в логе и `dlssvid info`; `--codec ffv1` и `--warp` работают везде.
- Ошибки FFmpeg/D3D12/CUDA — исключение `dlssvid::Error` с кодом и текстом; CLI печатает и возвращает 1.
