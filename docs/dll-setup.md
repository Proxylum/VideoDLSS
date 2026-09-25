# SDK и DLL NVIDIA: откуда брать, куда класть

> Установка из пакета (ZIP / инсталлер) и проверка golden-тестами — `docs/release.md`; CI — `docs/ci.md`.

Репозиторий не содержит бинарников NVIDIA и весов моделей (ТЗ §1, §9). Всё ниже пользователь
кладёт сам; сборка находит SDK только через переменные окружения.

## Переменные окружения

| Переменная | Что | Нужна с этапа |
|---|---|---|
| `VCPKG_ROOT` | корень vcpkg | 0 |
| `CUDA_PATH_V12_4` | CUDA Toolkit 12.4 (ставится инсталлятором CUDA) | 0 (interop), 2 (TensorRT) |
| `DLSS_SDK_ROOT` | клон [NVIDIA/DLSS](https://github.com/NVIDIA/DLSS) (NGX headers + `nvngx_dlss*.dll`) | 5 |
| `STREAMLINE_ROOT` | клон [NVIDIAGameWorks/Streamline](https://github.com/NVIDIAGameWorks/Streamline) — **не нужен**: FG идёт через NGX-API DLSS SDK (`docs/plans/07-fg.md`) | — |
| `NV_OPTICAL_FLOW_SDK_ROOT` | клон [NVIDIA/NVIDIAOpticalFlowSDK](https://github.com/NVIDIA/NVIDIAOpticalFlowSDK) (заголовки) | 3 |
| `TENSORRT_ROOT` | заголовки TensorRT 10.16: checkout `NVIDIA/TensorRT` тега `v10.16` (`include/`) или SDK zip. DLL (`nvinfer_10.dll`, `nvonnxparser_10.dll`) берутся в рантайме из `models/export/.venv/Lib/site-packages/tensorrt_libs` (pip `tensorrt-cu12==10.16.1.11`), либо из `DLSSVID_TENSORRT_DIR` / `TENSORRT_ROOT/lib`, либо из `bin\tensorrt` рядом с exe — так их кладёт полный пакет (`scripts\package.cmd full`, `docs/release.md`) | 2 |
| `DLSSVID_PYTHON` | интерпретатор для `depth_worker` и ONNX-экспорта (по умолчанию `models/export/.venv/Scripts/python.exe`) | 2 |
| `VDA_REPO` | checkout `DepthAnything/Video-Depth-Anything` (код VDA не является pip-пакетом; без переменной ищется в `models\export\third_party\Video-Depth-Anything`) | 2 |
| `HF_TOKEN` | необязательно: токен HuggingFace для быстрой загрузки весов | 2 |
| `QT_ROOT` / `CMAKE_PREFIX_PATH` | Qt 6.8 msvc2022_64 | 4 |
| `DLSSNR_PATCHER_ROOT` | клон [dev-camo/dlssnr-patcher](https://github.com/dev-camo/dlssnr-patcher) (`dlssnr_patcher.py`; или `--patcher`, или `tools/dlssnr-patcher` рядом с exe) | 6 |
| `CUDA_PATH_V13_3` | CUDA Toolkit 13.3 (`bin/ptxas`, `fatbinary`, `cuobjdump` для патчера; или `--cuda-bin`) — отдельно от CUDA 12.4 сборки | 6 |

Пример раскладки (`<SDK>` — общий каталог для SDK и клонов): `<SDK>\DLSS`, `<SDK>\Streamline`, `<SDK>\OpticalFlowSDK`,
`<SDK>\Qt\6.8.3\msvc2022_64`, `<SDK>\dlssnr-patcher`, `<SDK>\TensorRT` (заголовки v10.16),
`<SDK>\models\Depth-Anything-3`, `<SDK>\models\Video-Depth-Anything`.

## Модели глубины (этап 2)

Веса не входят в репозиторий: `models/registry.json` перечисляет id, источник на HuggingFace и лицензию;
`models/export/fetch.py` (или первый запуск `dlssvid depth`) скачивает их в `models/cache/`, экспортирует ONNX под
геометрию входа и собирает TensorRT-engine (`*.engine`, кэш рядом). Лицензии: DA3 metric/mono и VDA-Small —
Apache-2.0; VDA-Large — CC-BY-NC-4.0 (только исследования). ICDepth — публичного кода нет (2026-09-18).

## DLL в `bin/nvidia/`

| Файл | Откуда | Для чего |
|---|---|---|
| `nvngx_dlss.dll` | DLSS SDK `lib/Windows_x86_64/rel/` (сборка копирует её в `build/<preset>/bin/nvidia/`, если задан `DLSS_SDK_ROOT`) | DLSS Super Resolution (этап 5): `dlssvid upscale --backend dlss` |
| `nvngx_dlssg.dll` | DLSS SDK `lib/Windows_x86_64/rel/` (сборка копирует её в `bin/nvidia/` при `DLSS_SDK_ROOT`; в драйвере есть своя копия) | DLSS Frame Generation (этап 7) через NGX — без Streamline и `sl.*.dll` |
| `nvngx_dlssnr.dll` | из драйвера/игры с DLSS 5 (официально только RTX 50) | Neural Rendering (этап 6) |
| `nvngx_dlssnr.dll` (пропатченная) | результат `dlssnr-patcher` над вашей копией (`dlssvid nr-patch`; рядом сайдкар `nvngx_dlssnr.dll.patch.json`) | NR на RTX 20/30/40 |
| `nvngx.dll_dlssvid.dll` | собирается с проектом (цель `dlssvid_nr_forwarder`), лежит в `bin/` рядом с exe | форвардер: модуль, из которого вызывается `nvngx_dlssnr.dll` (этап 6) |

Приложение ищет DLL в `bin/nvidia/` рядом с exe (переопределяется `DLSSVID_NVIDIA_DLL_DIR` или `--dll-dir`). Свои DLL
(например, `nvngx_dlssnr.dll`) кладите в `bin/nvidia/` **дерева исходников** (git-ignored): сборка копирует оттуда все
`*.dll` в `build/<preset>/bin/nvidia/`, поэтому чистая пересборка их не теряет (с 2026-09-20). При
отсутствии DLL соответствующая стадия отключается с сообщением; приложение не падает: `upscale` переходит на
NIS (`--no-fallback` — ошибка вместо перехода). NGX пишет свой лог в `%LOCALAPPDATA%\dlssvid\ngx\`.

## RTX Video SDK (бэкенд `rtxvsr`) — удалён

Бэкенд `rtxvsr` был заглушкой: RTX Video SDK 1.1 выдаётся только под аккаунтом NVIDIA Developer, которого нет, и с
2026-09-20 он не был основным. 2026-09-24 (TASK-0021) заглушка удалена из фабрики, схемы параметров и CLI;
`RTX_VIDEO_SDK_ROOT` больше не нужен. Открытые альтернативы DLSS SR — `docs/plans/05-upscale.md`, раздел «Открытые
апскейлеры вместо RTX VSR».

## Neural Rendering (этап 6): `nvngx_dlssnr.dll`, форвардер, патч для RTX 20/30/40

Стадия `nr` использует NGX Feature 18 из `nvngx_dlssnr.dll`. Приложение **не распространяет** DLL: официальная лежит в
драйвере / играх с DLSS 5 и работает только на RTX 50 (Blackwell); на Ada и старше она отвечает
`0xBAD00001 FAIL_FeatureNotSupported`, и пользователь патчит **свою** копию.

1. Драйвер ≥ **616.56**: стадия читает версию из DXGI (`dlssvid nr --check` печатает её) и отказывает на более старом
   (`--skip-driver-check` — попробовать всё равно).
2. Положите `nvngx_dlssnr.dll` в `bin/nvidia/` рядом с exe (или задайте `DLSSVID_NVIDIA_DLL_DIR` / `--dll-dir`). RTX 50 —
   на этом всё; RTX 20/30/40 — шаг 3.
3. Патч на вашей машине (dlssnr-patcher, GPLv2, внешний инструмент; нужны `ptxas`, `fatbinary`, `cuobjdump` из
   **CUDA Toolkit 13.3** — отдельная установка от CUDA 12.4, с которой собирается проект):

   ```bat
   git clone https://github.com/dev-camo/dlssnr-patcher <SDK>\dlssnr-patcher
   dlssvid nr-patch --input C:\dlls\nvngx_dlssnr.dll --patcher <SDK>\dlssnr-patcher --cuda-bin "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin"
   ```

   Результат — `bin/nvidia/nvngx_dlssnr.dll` и сайдкар `nvngx_dlssnr.dll.patch.json` (SHA-256 входа и результата,
   команда, дата). `--arch ada|ampere|turing|blackwell|all` (по умолчанию все четыре, как у патчера), `--dry-run` — только
   проверка. Кнопка **«Пропатчить DLL…»** в панели проекта GUI делает то же (спрашивает DLL и, если переменные окружения
   не заданы, папки патчера и CUDA).
4. Проверка: `dlssvid nr --check` — GPU и архитектура, драйвер, путь / размер / SHA-256 DLL, форвардер, результаты `Init_Ext`
   сниппета и `CreateFeature(18)`, подсказка при ошибке; `--json file` — то же в JSON. Те же строки стадия пишет в лог при
   каждом запуске (ТЗ §4).
5. Форвардер `nvngx.dll_dlssvid.dll` собирается с проектом и должен лежать рядом с exe: модель принимает вызовы только из
   модуля с `nvngx.dll` в имени (`FAIL_PlatformError` иначе) — см. `docs/architecture.md`, «Хаки».
6. Патченная DLL **не подписана**: антивирус (Defender, сторонние) может поместить её в карантин или блокировать
   `LoadLibrary` (стадия сообщает код Win32). Добавьте исключение для папки `bin/nvidia/` (Defender: Безопасность Windows →
   Защита от вирусов и угроз → Управление настройками → Исключения). Не используйте пропатченную DLL с программами под
   античит-защитой. Новый драйвер может сломать патч — тогда повторите шаг 3 с DLL из нового драйвера.
7. Без DLL / на неподходящем GPU стадия `nr` в CLI завершается с инструкцией (код 1), в пайплайне и GUI отключается с
   сообщением — приложение не падает (ТЗ §4). Известная проблема патченных DLL — мерцание: guides `depth_dlss`/`mv_dlss`
   обязательны, `--temporal` — опциональный фильтр (`docs/benchmarks.md`).
8. Проверенные пары «SHA-256 DLL + версия драйвера» (заполняется по мере тестов):

   | DLL (SHA-256) | Версия DLL | Драйвер | GPU | Результат |
   |---|---|---|---|---|
   | `8270b350cd82de5ce89806872cdd6b6a9249b80836b91bbeb3573470744cc206` (165 840 496 байт) | 310.8.0.0, Ada-патч сообщества (HuggingFace Bandukids/DLSS-Runtimes — только для тестов, в продукт не входит) | 616.92 | RTX 4070 Ti SUPER (Ada, sm_89) | 2026-09-19: `Init_Ext` ABI 0 (ComfyUI), capability-блок, `CreateFeature(18)` Success; 30 кадров 1440p с guides, 15.9 мс/кадр GPU; `EvaluateFeature` стабилен |

Готовые патченные DLL (HuggingFace, Discord) в продукт не включаются — только ориентир для тестов.

## Frame Generation (этап 7): `nvngx_dlssg.dll` через NGX

DLSS SDK 310.9 содержит прямой NGX-API Frame Generation (`NVSDK_NGX_Feature_FrameGeneration`, «DLSS-FG Programming
Guide» в `doc/`), поэтому Streamline, скрытый swapchain и отдельный воркер не нужны (`fg_worker/README.md`).

1. `nvngx_dlssg.dll` — из DLSS SDK (`lib/Windows_x86_64/rel/`); сборка копирует её в `bin/nvidia/` рядом с
   `nvngx_dlss.dll`, когда задан `DLSS_SDK_ROOT`. Драйвер поставляет свою копию, но стадия требует файл в `bin/nvidia/`
   (или `DLSSVID_NVIDIA_DLL_DIR` / `--dll-dir`), как и для DLSS SR.
2. GPU: RTX 40 (Ada) и новее; ×3/×4 (Multi Frame Generation) — по `DLSSG.MultiFrameCountMax` из capability-блока (на
   Ada = 1 → только ×2; на RTX 50 до 3 → ×4). Без поддержки стадия печатает инструкцию и советует `--backend rife`.
3. Проверка: `dlssvid fg --check` — GPU, драйвер, DLL + SHA-256, `FrameGeneration.Available`, `MultiFrameCountMax`,
   результат `CreateFeature` и формат backbuffer (`rgba16f`, при отказе — `rgba8`).
4. Baseline RIFE (`--backend rife`): ONNX `yuvraj108c/rife-onnx` (RIFE 4.9/4.8/4.7, MIT) скачивается реестром моделей в
   `models/cache/` при первом запуске (21 МБ), TensorRT-движок под размер кадра собирается один раз (1–3 мин) и
   кэшируется рядом; нужны TensorRT (`TENSORRT_ROOT`) и CUDA.
