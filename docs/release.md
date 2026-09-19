# Релиз: пакет, установка, проверка golden-тестами

## Сборка пакета

```bat
scripts\package.cmd
:: или вручную после сборки:
cmake --build --preset release
cpack --preset release            :: build\release\dlss-video-<версия>-win64.zip
```

CPack собирает **ZIP** (переносимая раскладка) всегда и **NSIS-инсталлер** дополнительно, когда в `PATH` есть
`makensis` (NSIS 3; на машине сборки его пока нет — ZIP является основным форматом).

## Что внутри

```
dlss-video/
  bin/                  dlssvid.exe, dlssvid-gui.exe, dlssvid_golden_tests.exe, DLL vcpkg (FFmpeg, OpenEXR, PNG, TIFF, spdlog),
                        Qt 6 (windeployqt: Qt6*.dll, platforms/, imageformats/, ...), dxcompiler.dll, dxil.dll,
                        nvngx.dll_dlssvid.dll (форвардер NR)
  bin/nvidia/           README.md — сюда пользователь кладёт nvngx_dlss.dll, nvngx_dlssg.dll, nvngx_dlssnr.dll (docs/dll-setup.md)
  models/registry.json  реестр моделей; models/export/ — скрипты экспорта ONNX (python, requirements.txt)
  depth_worker/         python-воркер глубины (ICDepth / PyTorch)
  tests/golden/         данные golden-тестов (expected.json, ref/)
  docs/, README.md
```

Не входит (и не должно входить): бинарники NVIDIA (`nvngx_*.dll` из SDK, драйвера или игр), веса моделей и
TensorRT-движки (`models/cache/` заполняется при первом запуске), `*.pdb`, тесты unit/integration/app.

## Установка на чистой машине

1. Windows 10/11 x64, NVIDIA RTX 40/50, драйвер ≥ 616.56, установленный [CUDA Toolkit 12.4](https://developer.nvidia.com/cuda-12-4-0-download-archive)
   (рантайм `cudart64_12.dll`, NVDEC/NVENC через драйвер) и TensorRT 10.16 (`nvinfer_10.dll`, `nvonnxparser_10.dll` в
   `PATH` или рядом с exe — `docs/dll-setup.md`), Python 3.12 для экспорта моделей и `depth_worker`
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

## Полный прогон

```bat
bin\dlssvid.exe process -i clip.mp4 -o clip_result.mp4                     :: depth -> flow -> upscale x2 -> nr -> fg x2 -> encode + аудио
bin\dlssvid.exe process --project clip.dlssvid.json                       :: стадии и параметры из проекта (как в GUI)
bin\dlssvid.exe bench -i clip.mp4 --frames 30                             :: мс/кадр по стадиям
```
