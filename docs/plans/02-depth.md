# План этапа 2 — глубина

Дата: 2026-09-18. Ветка: `stage/02-depth`. Требования: ТЗ §2 (модели), §4 (стадия Depth), §8 (этап 2), §9.

## Цель
Пасс `depth_raw` (+ `depth_dlss`) для клипа тремя бэкендами за общим интерфейсом `IDepthEstimator`;
инференс DA3 и VDA через TensorRT FP16 из ONNX; PyTorch-путь как reference; temporal-стабилизация;
метрика TAE в логе; ICDepth — режим «Повышенное качество» в отдельном Python-воркере.

## Разведка (2026-09-18)
| Что | Статус | Решение |
|---|---|---|
| Depth Anything 3 | код Apache-2.0; веса: `DA3-*` (any-view, DualDPT + камера) — **CC-BY-NC-4.0**; `DA3METRIC-LARGE`, `DA3MONO-LARGE` — **Apache-2.0**, чистый DINOv2-L + DPT, прямое предсказание глубины | **Основной бэкенд: `DA3METRIC-LARGE`** (metric, метры) и `DA3MONO-LARGE` (relative). Экспорт в ONNX прост (одиночный кадр). Any-view DA3 не используем (NC-лицензия, не нужен для моно-видео) |
| Video Depth Anything | код Apache-2.0; `Metric-Video-Depth-Anything-{S,B,L}`, `Video-Depth-Anything-{S,L}`; окно 32 кадра, temporal-модуль | **Fallback / быстрый: `Metric-VDA-Small`** по умолчанию (лицензия и VRAM 16 ГБ), Large — опционально. ONNX-экспорт temporal-модели (окно 32×518) — пробуем; при неудаче VDA идёт через PyTorch-воркер, что фиксируется как отклонение от ТЗ |
| ICDepth (arXiv 2607.01677) | статья есть, **публичного кода и весов нет** (GitHub-поиск пуст) | Бэкенд `icdepth` в воркере объявлен, но помечен `unavailable`; режим «Повышенное качество» в v1 **заблокирован до выхода кода**. Воркер и протокол реализуются и проверяются на reference-путях DA3/VDA (PyTorch) |
| StableDPT | веса не подтверждены | вне v1 (ТЗ допускает) |
| TensorRT | pip `tensorrt-cu12==10.16.1.11` (DLL) + заголовки OSS `NVIDIA/TensorRT` (`D:\SDK\TensorRT\include`) | import-библиотеки генерируются из DLL на этапе конфигурации (`dumpbin`/`lib`); ONNX-парсер — `nvonnxparser_10.dll` из того же пакета |
| PyTorch | 2.14+cu126 (драйвер 591.86 ≥ 560 — ок) | venv `models/export/.venv`; общий и для `depth_worker` |
| Драйвер / VRAM | 591.86, 16 ГБ | VDA-L батч 32×518 ≈ 24 ГБ → по умолчанию Small; вход даунскейлится до ≤ 1080p (ТЗ §2) |

## Что делаем
1. **`core/ml/TrtEngine`** — обёртка TensorRT 10: логгер → spdlog, сборка engine из ONNX (FP16, фиксированные
   формы) с кэшем `models/cache/<model>.<gpu>.<trt>.engine`, выполнение на CUDA-буферах (cudart), синхронно.
   Импорт-библиотеки `nvinfer_10.lib` / `nvonnxparser_10.lib` генерируются CMake-скриптом из DLL pip-пакета
   (`TENSORRT_ROOT` или авто-поиск в venv). Без TensorRT сборка идёт с `DLSSVID_WITH_TENSORRT=OFF`.
2. **`core/ml/ModelRegistry`** — `models/registry.json` (id, url, sha256, license, входные параметры),
   загрузка в `models/cache/` при первом обращении (WinHTTP/urlmon), проверка SHA-256.
3. **`core/stages/depth/`**
   - `IDepthEstimator`: `Init(config)`, `WindowSize()`, `Estimate(span<const PassImage> rgb, span<PassImage> depth)`,
     `IsMetric()`, `ModelName()`; результат — `depth_raw` float32 (метры или relative с флагом).
   - `Da3Estimator` (TensorRT, кадр за кадром), `VdaEstimator` (TensorRT, окно 32 с перекрытием — если экспорт
     удался), `WorkerDepthEstimator` (клиент `depth_worker`: PyTorch DA3/VDA, ICDepth когда появится).
   - Препроцессинг: resize (bilinear) до `input_size` кратного 14 с сохранением пропорций, ImageNet-нормализация,
     NCHW; постпроцессинг: апсемпл до исходного разрешения edge-aware (guided filter по цвету), заполнение дыр.
   - `DepthPostProcess`: temporal scale-shift alignment по окну ±8 кадров (least squares к опорному кадру),
     edge-aware сглаживание, заполнение невалидных значений.
   - `DepthStage : IStage`: копит окно кадров, вызывает оценщик, пишет `depth_raw` (и `depth_dlss` через
     `DepthConvert`) в папки пассов и в слот `GpuFrameCache` (текстура R32F).
   - Метрика **TAE**: до появления MV (этап 3) — среднее |d_t − d_{t−1}| / mean(d) после выравнивания масштаба;
     пишется в лог и в `stage_params` манифеста; в этапе 3 заменяется на warp-версию.
4. **`depth_worker/`** — `worker.py` (JSON-lines по stdin/stdout, кадры через папки пассов: вход `color_source`
   PNG16/EXR, выход `depth_raw` NPZ), бэкенды `da3` (reference PyTorch), `vda` (PyTorch), `icdepth` (заглушка
   «недоступен»), `stub` (синтетика для тестов без GPU); `requirements.txt` → тот же venv.
5. **`models/export/`** — `export_da3.py` (safetensors → ONNX opset 17, проверка ORT vs torch), `export_vda.py`,
   `fetch.py` (HF → `models/cache`, sha256 в registry), pytest на экспорт.
6. **CLI:** `dlssvid depth -i clip -o passes --backend da3|vda|worker [--model id] [--range] [--no-stabilize]
   [--dlss --near --far] [--input-size 518] [--max-res 1080]`; `dlssvid models list|fetch <id>`.
7. **Тесты:** препроцессинг (геометрия, нормализация), scale-shift alignment на синтетике, guided upsample,
   TAE, воркер со `stub` (протокол, ошибки, падение процесса), `TrtEngine` на крошечной ONNX (`tests/data/tiny.onnx`,
   пропуск без GPU/TensorRT), `DepthStage` со `stub`-оценщиком (пассы, манифест, GPU-текстура), CLI как процесс;
   реальные модели — smoke на клипе из этапа 0 (в отчёте, не в ctest).

## Отложено / вне этапа
- NVDEC → CUDA → D3D12 без CPU переносится в **этап 3**: аппаратный OFA работает на CUDA-кадрах, путь делается один раз для обоих.
- Сравнение DA3 vs ICDepth в 2x2 — после этапа 4 (вьюпорт) и при появлении кода ICDepth.
- ONNX Runtime как fallback-рантайм — при необходимости в этапе 3 (одна обёртка на TRT/ORT).

## Приёмка
- `depth_raw`/`depth_dlss` на тестовом клипе бэкендами `da3` (TensorRT) и `vda` (TensorRT или воркер) и
  `worker` (PyTorch reference); TAE в логе; `ctest` зелёный без GPU (пропуски помечены).
- Отклонения от ТЗ явно записаны: ICDepth недоступен; VDA через TRT — если экспорт удался.
