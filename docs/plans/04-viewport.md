# План этапа 4 — вьюпорт

Дата: 2026-09-18. Ветка: `stage/04-viewport`. Требования: ТЗ §6 (превью и UI), §7 (Qt 6.8 Widgets + QWindow с D3D12,
HLSL 6.6 через DXC, QDockWidget), §8 (этап 4: все режимы §6 на кэше пассов этапов 2–3), §9 (без пересчёта,
≥ 30 fps скраббинга на 1080p).

## Разведка (2026-09-18)
| Что | Статус | Решение |
|---|---|---|
| Qt 6.8.3 | `<SDK>\Qt\6.8.3\msvc2022_64` (aqtinstall), собран MSVC 14.39; наш toolset 14.51 — ABI-совместим; есть `windeployqt`, плагины `qwindows`, `qoffscreen` | `find_package(Qt6 COMPONENTS Widgets)` через `CMAKE_PREFIX_PATH`/`QT_ROOT`; сборка GUI — опция `DLSSVID_BUILD_APP` (OFF без Qt); тесты Qt-логики — `QT_QPA_PLATFORM=offscreen` |
| DXC | Windows SDK 10.0.26100 `dxc.exe` 1.8; vcpkg `directx-dxc` даёт `dxc.exe` + `dxcompiler.dll` и `DIRECTX_DXC_TOOL` | шейдеры компилируются на этапе сборки (`dxc -T ps_6_0 … -Fh`) в заголовки и встраиваются в exe; целевая модель SM 6.0 (работает на WARP для тестов), 6.6 не требуется |
| Композитинг поверх нативного окна | Qt не рисует виджеты поверх дочернего нативного окна со swapchain | вьюпорт — `QWindow` в `createWindowContainer` со своим DXGI-swapchain на общем `D3D12Device`; подписи ячеек и пробник — виджеты Qt рядом с вьюпортом; в PNG-скриншот подписи вписывает `QPainter` |

## Что делаем
1. **`core/viewport/`** (без Qt, тестируется на WARP):
   - `ViewportState` — модель состояния (JSON): режим `single|overlay|grid`, стек слоёв (`source` = пасс/`source`/`result`,
     display: `color|grayscale|viridis|turbo|mv_hsv|mv_arrows|mv_magnitude|mask_fill|mask_contour`, min/max/auto/invert,
     opacity, blend `normal|difference|multiply|screen`, visible, solo), источники 4 ячеек и развёрнутая ячейка,
     wipe (ось, позиция, пара слоёв), трансформация вида (zoom 25–800 %, pan, fit), текущий кадр.
   - `Composite.hlsl` — **один** пиксельный шейдер: до 5 слоёв, колормапы (grayscale, viridis, turbo — полиномы),
     MV как HSV/magnitude, маски заливкой/контуром, blend-режимы, wipe, фон вне кадра; `Arrows.hlsl` —
     стрелки MV по сетке (инстансированные линии, VS читает текстуру MV, без CPU).
   - `ViewportRenderer` — D3D12 на общем девайсе: root signature (CBV + таблица 5 SRV, статические сэмплеры),
     PSO композитора и стрелок, swapchain для HWND и offscreen-цель для скриншотов/тестов, 2x2 — четыре draw
     с viewport/scissor ячейки и общей трансформацией, `ReadbackTexel` для пробника (1×1 readback, без копий кадра).
   - `FrameStore` — GPU-кэш кадров пассов: обнаружение папок пассов (`manifest.json`), фоновый загрузчик
     (EXR/PNG/… → CPU → upload на render-потоке с бюджетом), декод исходного видео с seek (`VideoDecoder::SeekToFrame`),
     префетч ±N кадров, LRU-вытеснение по бюджету VRAM, min/max пасса для авто-диапазона, статусы «нет / грузится / готов».
   - `Project` — файл проекта `*.dlssvid.json`: исходник, корень пассов, найденные пассы, результат, состояние
     вьюпорта, конфиги стадий (enabled + params).
2. **`app/`** (Qt 6.8 Widgets): `MainWindow` с доками — слева `ProjectPanel` (источник, список пассов со статусом,
   стадии с включателями/параметрами и «Запустить» → `dlssvid depth|flow` в `QProcess`), центр — `ViewportWindow`
   (QWindow + swapchain; колесо — zoom к курсору 25–800 %, средняя кнопка — pan, `F` — вписать, `1:1`, `1…9` —
   источник, клик по ячейке 2x2 — развернуть, перетаскивание шторки) + `TimelineWidget` (скраббер, номер кадра,
   play/pause, ±1; FG-результат — ×2 fps; play только по готовым кадрам), справа `InspectorPanel` (слой: колормап,
   min/max/auto, инверсия, режим MV/масок, прозрачность слайдер + число, blend, видимость, соло; wipe; источники
   ячеек), внизу `LogPanel` (spdlog → Qt) и `TaskQueuePanel` (прогресс задач). Пробник — координаты и сырые значения
   всех пассов под курсором (readback 1×1). Скриншот 2x2 → PNG (offscreen-рендер + подписи). Состояние вьюпорта
   сохраняется в проект.
3. **CLI-паритет:** `dlssvid project init -i clip.mp4 --passes dir -o proj.dlssvid.json`, `dlssvid render --project
   proj.json --frame N [--mode …] -o shot.png` — offscreen-рендер с сохранённым состоянием (тесты, скриншоты без GUI).
4. **Тесты (WARP):** JSON round-trip состояния/проекта; рендер: grayscale ровно по формуле, viridis/turbo
   монотонны по яркости и с известными концами, blend-режимы по формулам, opacity, wipe, 2x2 раскладка и развёрнутая
   ячейка, zoom/pan (пиксель под курсором), MV-HSV для известных векторов, маска заливкой; FrameStore: обнаружение
   пассов, загрузка/статусы, LRU при малом бюджете, префетч; пробник; CLI `render`; бенчмарк композита 1080p
   (в отчёте, ≥ 30 fps). Qt: `AppModel`/`TaskQueue` в offscreen QPA.

## Отложено / вне этапа
- GPU-препроцессинг моделей (этап 5 использует ту же шейдерную инфраструктуру).
- Стадии SR/NR/FG в панели стадий появляются по мере этапов 5–7 (панель уже умеет запускать любую CLI-команду).

## Приёмка (ТЗ §8–9)
- Все режимы §6 работают на кэше пассов этапов 2–3 без пересчёта; ≥ 30 fps скраббинга на 1080p (замер в отчёте);
  состояние вьюпорта в проекте; скриншот 2x2 в PNG; всё доступно из CLI (`render`).

## Итог (2026-09-18)

- `core/viewport/`: `ViewportState` (JSON), `Composite.hlsl` + `Arrows.hlsl` (DXC → заголовки при сборке, SM 6.0),
  `ViewportRenderer` (swapchain / offscreen, 2x2, пробник), `FrameStore` (2–4 loader-потока, префетч ±6, LRU по бюджету
  1.5 ГиБ, кадры «по пути» при seek), `Project` (`*.dlssvid.json`, относительные пути). `VideoDecoder`: `SeekToFrame`,
  `SeekToKeyframeBefore`, потоки декодера; `ColorConvert`: `Yuv420pToRgba16f` (F16C), `ToRgba16f`.
- `app/`: `dlssvid-gui` (Qt 6.8 Widgets, `QWindow` + DXGI swapchain на общем девайсе) — панели проекта/инспектора/
  лога/задач, таймлайн, хоткеи, скриншот PNG с подписями; стадии запускаются через `TaskQueue` (CLI-процесс).
- CLI: `dlssvid project init|show`, `dlssvid render` (single/overlay/grid, слои, wipe, zoom, `--bench`, `--save-state`).
- Тесты: +5 файлов (`test_viewport_state`, `test_viewport_renderer` — WARP, `test_frame_store`, `test_project`,
  `test_cli_render` + `test_viewport_sources`), `tests/app/test_app_model` (offscreen QPA).
- Приёмка: все режимы §6 на кэше пассов этапов 2–3 (smoke `build/release/smoke/in.dlssvid.json`); скраббинг 1080p —
  один источник 82 fps холодный / > 1000 fps тёплый, сетка 2x2 из 4 источников 37 fps холодный; состояние в проекте;
  скриншот 2x2 в PNG; всё из CLI.
- Отложено: NVDEC-кольцо и YUV→RGB на GPU (этап 5), несколько кадров в полёте у рендерера, `mv_arrows` для
  разреженных полей на больших зумах (шаг стрелок фиксирован в image px).
