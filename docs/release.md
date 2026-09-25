# Релиз: пакет, установка, проверка golden-тестами

## Сборка пакета

```bat
scripts\package.cmd               :: build\release\dlss-video-<версия>-win64.zip — то, что собирает CI
scripts\package.cmd full          :: build\release\dlss-video-<версия>-win64-full.zip — самодостаточный (см. ниже)
:: или вручную после сборки:
cmake --preset release -DDLSSVID_PACKAGE_FULL=ON
cmake --build --preset release
cpack --preset release
```

CPack собирает **ZIP** (переносимая раскладка) всегда и **NSIS-инсталлер** дополнительно, когда в `PATH` есть
`makensis` (NSIS 3; на машине сборки его пока нет — ZIP является основным форматом).

## Что внутри

```
dlss-video-<версия>-win64/
  bin/                  dlssvid.exe, dlssvid-gui.exe, dlssvid_golden_tests.exe, DLL vcpkg (FFmpeg, OpenEXR, PNG, TIFF, spdlog),
                        Qt 6 (windeployqt: Qt6*.dll, platforms/, imageformats/, ...), dxcompiler.dll, dxil.dll,
                        nvngx.dll_dlssvid.dll (форвардер NR)
  bin/nvidia/           README.md — сюда пользователь кладёт nvngx_dlss.dll, nvngx_dlssg.dll, nvngx_dlssnr.dll (docs/dll-setup.md)
  models/registry.json  реестр моделей; models/export/ — скрипты экспорта ONNX (python, requirements.txt)
  depth_worker/         python-воркер глубины (ICDepth / PyTorch)
  sr_worker/            python-воркер видео-апскейла (RealBasicVSR)
  tests/golden/         данные golden-тестов (expected.json, ref/)
  docs/, README.md, CHANGELOG.md, LICENSE
```

Не входит (и не должно входить): бинарники NVIDIA (`nvngx_*.dll` из SDK, драйвера или игр), исходные веса моделей
(safetensors/pth) и TensorRT-движки (собираются при первом запуске в `models/cache/`), `*.pdb`, тесты unit/integration/app.

### Полный пакет (`package.cmd full`, `-DDLSSVID_PACKAGE_FULL=ON`)

Тот же состав плюс всё, без чего пайплайн не заработает на чистой машине без CUDA Toolkit, TensorRT и Python:

```
  bin/tensorrt/         рантайм TensorRT 10.16: nvinfer_10.dll, nvonnxparser_10.dll, nvinfer_plugin_10.dll,
                        nvinfer_builder_resource_{ptx,sm75,sm86,sm89,sm120}_10.dll (RTX 20/30/40/50; список — DLSSVID_PACKAGE_TRT_SMS)
  bin/cudart64_12.dll   рантайм CUDA 12, который импортируют dlssvid.exe (interop)
  models/cache/*.onnx   экспортированные модели: DA3 metric large, Metric-VDA small (16:9 518×924 и 2.4:1 378×910),
                        SEA-RAFT (1280×720, 1728×720), RIFE 4.9 — всё, что лежит в models/cache на машине сборки
```

Источники по умолчанию — `DLSSVID_TENSORRT_DIR` (или venv `models/export/.venv/.../tensorrt_libs`), `CUDA_PATH`,
`DLSSVID_MODELS_DIR/cache` (или `models/cache`); переопределяются кэш-переменными `DLSSVID_PACKAGE_TENSORRT_DIR`,
`DLSSVID_PACKAGE_MODELS_CACHE`. Размер — несколько ГБ (TensorRT ≈ 2 ГБ, ONNX ≈ 5 ГБ), поэтому CI собирает только
обычный пакет. Загрузчик ищет TensorRT в `bin\tensorrt` рядом с exe (после `DLSSVID_TENSORRT_DIR`/`TENSORRT_ROOT`),
модели — в `models/cache` рядом с `registry.json`. TensorRT-движки под конкретную GPU собираются при первом запуске
(DA3 ≈ 2 мин, VDA ≈ 4 мин, SEA-RAFT ≈ 2.5 мин) и кладутся в тот же `models/cache` — папка должна быть доступна
на запись (распаковывайте не в Program Files, либо задайте `DLSSVID_MODELS_DIR`); глубина папки не важна — имена
движков длинные, и пути за 260 символов открываются через `\\?\` (`core/util/Paths.h`, с 0.2.0). ONNX другого соотношения сторон
(не 16:9 и не 2.4:1) по-прежнему требует экспорта через venv (`docs/dll-setup.md`).

## Установка на чистой машине

1. Windows 10/11 x64, NVIDIA RTX 40/50, драйвер ≥ 616.56. С **полным** пакетом больше ничего не нужно (рантайм CUDA,
   TensorRT и модели внутри); с обычным — установленный [CUDA Toolkit 12.4](https://developer.nvidia.com/cuda-12-4-0-download-archive)
   (рантайм `cudart64_12.dll`) и TensorRT 10.16 (`nvinfer_10.dll`, `nvonnxparser_10.dll` в `PATH`, `bin\tensorrt` или
   `DLSSVID_TENSORRT_DIR` — `docs/dll-setup.md`), Python 3.12 для экспорта моделей и `depth_worker`
   (`models/export/requirements.txt`).
2. Распаковать ZIP (или запустить инсталлер).
3. Положить DLL NVIDIA в `bin\nvidia\` (`docs/dll-setup.md`): `nvngx_dlss.dll`, `nvngx_dlssg.dll` из DLSS SDK,
   `nvngx_dlssnr.dll` — своя копия (RTX 40: пропатченная, `dlssvid nr-patch`).
4. Проверка окружения: `bin\dlssvid.exe info`, `bin\dlssvid.exe nr --check`, `bin\dlssvid.exe fg --check`.

## Golden-тесты на установленной копии

```bat
set DLSSVID_CLI=C:\dlss-video\bin\dlssvid.exe
set DLSSVID_GOLDEN_DIR=C:\dlss-video\tests\golden
C:\dlss-video\bin\dlssvid_golden_tests.exe            :: детерминированный пайплайн (WARP) + реальные бэкенды на GPU
C:\dlss-video\bin\dlssvid_golden_tests.exe "[golden]~[gpu]"   :: только детерминированная часть
```

Эталоны (`tests/golden/ref/`) перезаписываются разработчиком командой `set DLSSVID_GOLDEN_UPDATE=1` перед запуском
теста из build-дерева — только когда изменение результата осознанно (новая версия NIS, тонмаппинга и т. п.).

## Выпуск версии

1. Ветка `release/X.Y.Z`: `project(dlss-video VERSION X.Y.Z)` в `CMakeLists.txt`, раздел в `CHANGELOG.md`, строка версии в README.
   Версия релиза не входит в отпечатки пассов: поле `tool.app` — версия алгоритма пассов `kPassToolVersion`
   (`core/pipeline/PassFingerprint.h`), её поднимают отдельно и только когда результат стадии меняется при тех же
   параметрах, иначе релиз обесценил бы все пассы пользователей.
2. Локально: сборка, `ctest`, `scripts\package.cmd` (и `full`), распаковать ZIP во временную папку, golden на распакованной
   копии (`DLSSVID_CLI`, `DLSSVID_GOLDEN_DIR`; для `[gpu]`-части положить DLL NVIDIA в её `bin\nvidia`).
3. MR → CI → слияние в `main` → аннотированный тег `vX.Y.Z` на merge-коммите → push тега: пайплайн тега собирает
   `dlss-video-X.Y.Z-win64.zip` (job `package`, артефакт хранится месяц) → GitLab Release на теге (API `releases`) с разделом
   CHANGELOG и ссылкой на артефакт `https://<хост GitLab>/<группа>/<проект>/-/jobs/artifacts/vX.Y.Z/download?job=package`.
4. Полный пакет (несколько ГБ) в CI не собирается — собирается на машине разработчика и выкладывается вручную.

## Полный прогон

```bat
bin\dlssvid.exe process -i clip.mp4 -o clip_result.mp4                     :: depth -> flow -> upscale x2 -> nr -> fg x2 -> encode + аудио
bin\dlssvid.exe process --project clip.dlssvid.json                       :: стадии и параметры из проекта (как в GUI)
bin\dlssvid.exe bench -i clip.mp4 --frames 30                             :: мс/кадр по стадиям
```
