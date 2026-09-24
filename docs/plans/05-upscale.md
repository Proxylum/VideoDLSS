# План этапа 5 — апскейл

Дата: 2026-09-18. Ветка: `stage/05-upscale`. Требования: ТЗ §3 (`IUpscaler`, RTX VSR по умолчанию, DLSS SR с
эмуляцией джиттера — HACK, масштабы ×1.5/×2/×3, потолок 4K, флаг «artifact reduction only»), §4 (стадия Upscale: цвет
+ depth/MV/jitter → цвет целевого разрешения, `color_sr`), §8 (этап 5), §9 (оба режима дают `color_sr`; A/B по 5 клипам,
PSNR/SSIM к 4K-даунскейл-эталону в `docs/benchmarks.md`).

## Разведка (2026-09-18)
| Что | Статус | Решение |
|---|---|---|
| RTX Video SDK 1.1 | **нет на машине** — требует аккаунт NVIDIA (TASK-0011, действие пользователя); публичного зеркала заголовков нет | бэкенд `rtxvsr` — точка интеграции за `IUpscaler` (опция `RTX_VIDEO_SDK_ROOT`); без SDK стадия сообщает инструкцию и **деградирует в `nis`** (по ТЗ §4: отсутствие SDK не роняет приложение); при появлении SDK — реализуется в отдельном MR |
| DLSS SDK 310.9.1 (`D:\SDK\DLSS`, NGX API 1.5) | заголовки, `nvsdk_ngx_d.lib`, `nvngx_dlss.dll` (rel) | `DLSS_SDK_ROOT` → `DLSSVID_WITH_DLSS`; DLL пользователь кладёт в `bin/nvidia/` (`docs/dll-setup.md`); на машине разработки копируется из SDK |
| NVIDIA Image Scaling 1.0.3 (MIT, GitHub) | склонирован в `D:\SDK\NVIDIAImageScaling`; шейдер — заголовок HLSL + таблицы коэффициентов | вендорим `NIS_Scaler.h`/`NIS_Config.h` в `third_party/nis/` (MIT), compute-PSO через DXC (`cs_6_0`, работает на WARP) — детерминированный бэкенд для тестов и baseline для A/B; режим NVSharpen (без апскейла) = аналог «artifact reduction only» |
| Требования NGX к DLSS SR (Programming Guide 310) | входы в `NON_PIXEL_SHADER_RESOURCE`, выход UAV; MV: `p(t−1) = p(t) + mv`, пиксели, y вниз — совпадает с `mv_dlss`; `MVLowRes`, если MV в разрешении входа; depth 0..1, `DepthInverted` для reverse-Z (= `depth_dlss`); LDR: 0..1, нелинейно (sRGB/BT.709 gamma) — наш декодированный цвет; jitter ∈ [−0.5, 0.5] в пикселях входа, тот же знак, что у MV; фаз = 8·(target/render)²; пресеты J/K/L/M | флаги `MVLowRes` (когда `mv_dlss` в разрешении источника) + `DepthInverted`; LDR-режим; Halton(2,3); `--preset` |

## Что делаем
1. **`core/stages/upscale/`**
   - `IUpscaler` — `Init(device, config{in, out, sharpness, preset, artifactReductionOnly})`, `Evaluate(cmdList, inputs{color RGBA16F, depth R32F?, mv RG32F?, jitter, reset}, output RGBA16F UAV)`, `Describe()`; фабрика по имени: `rtxvsr | dlss | nis | bicubic`.
   - `Jitter` — Halton(2,3) → смещения в [−0.5, 0.5), число фаз по формуле NGX.
   - `Resample.hlsl` (compute, Catmull-Rom) — субпиксельный сдвиг входа для эмуляции джиттера (**HACK**, ТЗ §3) и наивный бэкенд `bicubic` (baseline для A/B, WARP-тесты).
   - `NisUpscaler` — NVScaler (масштаб) / NVSharpen (artifact-reduction-only); таблицы коэффициентов как текстуры.
   - `DlssUpscaler` — NGX: `Init_with_ProjectID` (путь DLL `bin/nvidia/`, лог в spdlog), capability check (`SuperSampling.Available`, драйвер), `GET_OPTIMAL_SETTINGS`, `CREATE_DLSS_EXT` (флаги LDR/MVLowRes/DepthInverted, пресет), `EVALUATE_DLSS_EXT` с jitter/MV/depth/reset; выход через UAV-текстуру → копия в слот кэша.
   - `RtxVsrUpscaler` — заглушка с инструкцией (SDK 1.1, `RTX_VIDEO_SDK_ROOT`); интерфейс готов.
   - `UpscaleStage` — цвет кадра → RGBA16F на GPU; depth/MV из слота кэша (если стадии 2–3 в том же пайплайне) или из папок пассов (`--depth-dir`, `--mv-dir`); джиттер → апскейлер → `color_sr` (EXR half / PNG16) + текстура в слоте; опционально видео (`--video`, NVENC/ffv1) через `RgbToYuv420p`; статистика мс/кадр. Цель: `--scale 1.5|2|3` или `--target WxH`, потолок 3840×2160 с сохранением пропорций, чётные размеры.
2. **CLI**: `dlssvid upscale -i in.mp4 -o passes --backend rtxvsr|dlss|nis|bicubic --scale 2 [--target WxH] [--depth-dir] [--mv-dir] [--sharpness] [--preset] [--no-jitter] [--jitter-sign] [--artifact-reduction-only] [--format] [--video out.mp4 --codec] [--frames N] [--warp]`; `dlssvid compare --ref A --test B` (видео или папки пассов) → PSNR/SSIM (Y, RGB) по кадрам и средние, JSON — инструмент A/B для этого и следующих этапов.
3. **GUI/вьюпорт**: стадия `upscale` в `Project::DefaultStages()` (панель запускает `dlssvid upscale`); `color_sr` появляется как пасс (kind Color) — A/B в 2x2 и wipe уже работают, слои сэмплируются в своём разрешении.
4. **Тесты**: юнит — Halton, разрешение цели (потолок 4K, чётность), `RgbToYuv420p` round-trip, PSNR/SSIM на известных данных, resample на WARP (целочисленный сдвиг = точный сдвиг), NIS ×2 на WARP (размер, детерминизм, PSNR к эталону не хуже bicubic), стадия со стабом/`bicubic` на WARP (пасс `color_sr`, манифест, слот кэша); интеграционные — CLI `upscale` (nis) и `compare` как процессы, `--video`; DLSS — на NVIDIA GPU с `nvngx_dlss.dll` (SKIP иначе): ×2 на синтетическом 4K→1080p→4K, PSNR к эталону, выбор знака джиттера по PSNR.
5. **Документация**: `docs/benchmarks.md` (методика 4K → 1080p → ×2 → PSNR/SSIM, результаты на доступных клипах; 5 клипов ТЗ — после TASK-0010), `docs/architecture.md` (стадия, HACK-строки: эмуляция джиттера, VSR-заглушка), `docs/dll-setup.md` (`DLSS_SDK_ROOT`, `bin/nvidia/nvngx_dlss.dll`), README, `third_party/nis/` с лицензией.

## Открытые вопросы
- Знак джиттера при эмуляции (сдвиг содержимого +j ⇔ семплирование в p − j): руководство задаёт систему координат MV, но не даёт формулы для ресемплинга; выбирается по PSNR на синтетическом тесте, фиксируется опцией `--jitter-sign` и в `docs/benchmarks.md`.
- RTX VSR: реальная интеграция после получения SDK (TASK-0011); DoD «оба режима дают color_sr» в этом MR выполняется как «оба режима запускаются; `rtxvsr` без SDK деградирует в `nis` с инструкцией».
- A/B по 5 клипам ТЗ зависит от TASK-0010 (тестовые клипы); в этом этапе — методика, CLI и результаты на синтетическом 4K и smoke-клипе.

## Отложено / вне этапа
- RTX Video HDR (ТЗ §3: не в v1), автоэкспозиция/HDR-вход DLSS (вход SDR), детекция смены сцены для `Reset` (пока только первый кадр).

## Приёмка (ТЗ §8–9)
- `dlssvid upscale --backend dlss|nis` дают `color_sr` (×1.5/×2/×3, потолок 4K); `rtxvsr` — через SDK при наличии, иначе инструкция + fallback; A/B в 2x2/wipe во вьюпорте; PSNR/SSIM в `docs/benchmarks.md`; HACK описан в `docs/architecture.md`.

## Итог (2026-09-19)

- `core/stages/upscale/`: `IUpscaler` + фабрика (`rtxvsr | dlss | nis | bicubic`), `ResolveUpscaleTarget` (×1.5/×2/×3,
  явный размер, потолок 4K), `Jitter` (Halton(2,3), фазы по формуле NGX), `Resampler`/`BicubicUpscaler`
  (`Resample.hlsl`), `NisUpscaler` (NVScaler/NVSharpen, `third_party/nis`, до 2 проходов), `DlssUpscaler` (NGX D3D12:
  init, capability check, optimal settings, флаги, пресеты, evaluate), `RtxVsrUpscaler` (заглушка с инструкцией),
  `UpscaleStage` (`color_sr` EXR/PNG16, слот кэша, `--video`), `RunUpscale`; `gpu/ComputeKernel`, `passes/ImageMetrics`,
  `ColorConvert::RgbToYuv420p`.
- CLI `dlssvid upscale` (все опции ТЗ §3, fallback, `--no-fallback`), `dlssvid compare` (PSNR/SSIM, JSON); GUI: стадия
  `upscale` в проекте по умолчанию (`rtxvsr`, ×2); `color_sr` во вьюпорте.
- DLSS SR реально работает на RTX 4070 Ti SUPER (драйвер 591.86, SDK 310.9.1): знак эмулированного джиттера
  подтверждён тестом (+j 47.7 дБ vs 46.0 без джиттера vs 39.7 −j); 720p → 1440p с depth/MV guides — 11 мс GPU.
- Бенчмарк (`docs/benchmarks.md`): синтетика 360p→720p и 540p→1080p — bicubic/NIS/DLSS в пределах 0.7 дБ;
  RTX VSR не измерен (нет SDK), 5 клипов ТЗ — после TASK-0010.
- Тесты: +3 файла (`test_upscale` — 7 юнитов на WARP, `test_upscale_stage` — стадия, fallback, PNG16, потолок,
  слот, DLSS на GPU, `test_cli_upscale`); всего 140.
- Отложено: интеграция RTX Video SDK (TASK-0011), 5 клипов (TASK-0010), копия результата в слот без CPU, фоновая
  запись EXR, цветовые теги в превью-видео, детекция смены сцены для `Reset`.

## Открытые апскейлеры вместо RTX VSR (предложение 2026-09-24, TASK-0021)

Заглушка `rtxvsr` удалена: SDK закрыт аккаунтом, а пункт в списке методов только вводил в заблуждение. Ниже —
открытые модели, которые имеет смысл поставить рядом с DLSS SR как «максимальное качество без NGX». Скорость —
порядок величины для 1080p → 4K на RTX 4070 Ti (TensorRT FP16), уточняется бенчмарком на 5 клипах ТЗ.

| Модель | Лицензия | Тип | Качество на реальном видео | Скорость (оценка) | Интеграция в `IUpscaler` |
|---|---|---|---|---|---|
| **Real-ESRGAN** — `RealESRGAN_x2plus` / `x4plus` (RRDBNet), `realesr-general-x4v3` (компактная SRVGG) | BSD-3 (код и веса) | покадровый SISR | высокое, «дорисовывает» фактуру; на шумных исходниках — перерезкость и мерцание между кадрами | x2plus ≈ 0,3–0,6 с/кадр; general-x4v3 ≈ 20–40 мс/кадр | ONNX → TensorRT (чистые свёртки), тайлы с перекрытием как у глубины; ×1.5 и ×3 — через x2/x4 + `Resampler` |
| **RealBasicVSR** (BasicVSR++ для реального видео, MMagic) | Apache-2.0 | видео-SR: двунаправленное распространение по кадрам, выравнивание по потоку | лучшая временная стабильность и детализация среди открытых | ≈ 1–2 с/кадр в PyTorch | ONNX/TensorRT напрямую нет (DCNv2, рекуррентность) → Python-воркер на PyTorch по образцу `depth_worker`, окна 15–30 кадров |
| **SwinIR** (real-world GAN, ×2/×4) | Apache-2.0 | трансформер SISR | на уровне Real-ESRGAN, естественнее фактура | ≈ 1–3 с/кадр | ONNX с фиксированным окном, на 4K — только тайлами |
| **SPAN**, **RealPLKSR** (+ общественные веса Nomos/Phhofm) | Apache-2.0 / MIT; часть весов CC-BY-NC — проверять | лёгкий SISR | между NIS и Real-ESRGAN; RealPLKSR близок к Real-ESRGAN | 10–30 мс/кадр | ONNX → TensorRT, просто; быстрый режим |
| HAT / DAT / DRCT | MIT / Apache-2.0 | тяжёлые трансформеры SISR | максимум PSNR на бенчмарках bicubic-деградации; на сжатом видео артефакты | 3–10 с/кадр | непрактично для ×2 на 4K |
| Anime4K | MIT | шейдеры | только аниме | реальное время | вне темы |

**Рекомендация.**
1. Один общий бэкенд `trt` — «ONNX-модель через TensorRT» с реестром моделей как у глубины (`ModelRegistry`,
   engine-кэш, тайлы с перекрытием, FP16): по умолчанию `RealESRGAN_x2plus` (максимум качества покадрово, лицензия
   BSD-3), быстрый вариант `realesr-general-x4v3`, опционально RealPLKSR. Один MR: экспорт ONNX в `models/export/`,
   бэкенд, схема параметров (`model`, `tile`), тесты на WARP с tiny-моделью, бенчмарк `dlssvid compare` на 5 клипах.
2. Затем **RealBasicVSR** через PyTorch-воркер как режим «максимальное качество для видео» (временная стабильность,
   которой у покадровых моделей нет): 2–3 MR, окна кадров, кэш пассов как у остальных стадий.
3. NR после нейросетевого апскейла оставить выключаемым: Real-ESRGAN уже усиливает резкость, двойное усиление даёт
   ореолы.
