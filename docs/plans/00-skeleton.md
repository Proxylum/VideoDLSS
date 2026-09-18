# План этапа 0 — скелет

Дата: 2026-09-18. Ветка: `stage/00-skeleton`. Источник требований: ТЗ §7–§9.

## Цель
Видео проходит через приложение без изменений: `dlssvid process --passthrough` декодирует файл,
поднимает кадры на GPU (D3D12) и кодирует обратно; кадры побитово совпадают с FFmpeg-декодом.

## Что делаем
1. **Сборка:** CMake ≥ 3.28 + Ninja + vcpkg (manifest, baseline закреплён; FFmpeg 7.1.1 через override),
   пресеты `debug`/`release`, `scripts/build.cmd`. Пути к SDK — только через переменные окружения.
2. **`core/gpu`:** `D3D12Device` (адаптер high-performance или WARP, direct queue, fence, синхронные
   upload/readback буферов и текстур с учётом выравнивания row pitch); `GpuFrameCache` — кольцевой
   буфер слотов; `CudaInterop` — импорт D3D12-буфера в CUDA через external memory, выбор CUDA-девайса по LUID.
3. **`core/io`:** `VideoDecoder` (FFmpeg, опционально NVDEC через hwaccel cuda с bit-exact NV12→YUV420P,
   аудио-пакеты в sink), `VideoEncoder` (NVENC по умолчанию, `ffv1` для lossless-тестов, stream copy аудио).
4. **`core/pipeline`:** `Frame` (YUV420P, tightly packed), `IStage`/`StageConfig`(JSON)/`FrameContext`,
   `Pipeline` (линейный список стадий), `RunPipeline` (decode → stages → encode).
5. **`stages/passthrough`:** загрузка кадра в слот кэша и чтение обратно с проверкой bit-exact.
6. **CLI:** `dlssvid info`, `dlssvid process --passthrough [--codec] [--hwaccel] [--frames] [--warp]`.
7. **Тесты:** юнит (геометрия кадра, D3D12 round-trip, кэш, порядок стадий, CUDA interop) и
   интеграционные (синтетический ffv1-клип → passthrough → побитовое сравнение; аудио stream copy;
   нечётные размеры; NVDEC = software decode; CLI как процесс).

## Открытые вопросы и принятые решения
- **Bit-exact на каком уровне?** На уровне декодированных плоскостей YUV 8-бит 4:2:0 (кодирование
  всегда lossy, кроме ffv1). Сравнение RGB отложено: конверсия YUV→RGB будет на GPU в этапе 2+.
- **Кадры на CPU в этапе 0.** ТЗ требует NVDEC с CUDA↔D3D12 interop без копий через CPU; в этапе 0
  interop реализован и протестирован на буферах, но кадры декодера идут через CPU (`av_hwframe_transfer_data`).
  Прямой путь NVDEC → CUDA → D3D12 без CPU — задача этапа 2 (вместе с TensorRT), см. `docs/architecture.md`.
- **>8-бит источники** конвертируются swscale в yuv420p с предупреждением; high bit depth сохраняется с этапа 2.
- **FFmpeg 7.1.1**, а не 9.x из текущего vcpkg: по ТЗ; override в `vcpkg.json`.
- **CI** не настраивается (решение брифинга); тесты запускаются локально `ctest --preset release`.

## Приёмка
- `ctest --preset release` зелёный на машине с RTX; GPU-специфичные тесты пропускаются без NVIDIA.
- `dlssvid process --passthrough --codec ffv1` даёт файл, декод которого побитово равен декоду входа.
