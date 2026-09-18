# SDK и DLL NVIDIA: откуда брать, куда класть

Репозиторий не содержит бинарников NVIDIA и весов моделей (ТЗ §1, §9). Всё ниже пользователь
кладёт сам; сборка находит SDK только через переменные окружения.

## Переменные окружения

| Переменная | Что | Нужна с этапа |
|---|---|---|
| `VCPKG_ROOT` | корень vcpkg | 0 |
| `CUDA_PATH_V12_4` | CUDA Toolkit 12.4 (ставится инсталлятором CUDA) | 0 (interop), 2 (TensorRT) |
| `DLSS_SDK_ROOT` | клон [NVIDIA/DLSS](https://github.com/NVIDIA/DLSS) (NGX headers + `nvngx_dlss*.dll`) | 5 |
| `STREAMLINE_ROOT` | клон [NVIDIAGameWorks/Streamline](https://github.com/NVIDIAGameWorks/Streamline) | 7 |
| `RTX_VIDEO_SDK_ROOT` | RTX Video SDK 1.1 (developer.nvidia.com, требует аккаунт) | 5 |
| `NV_OPTICAL_FLOW_SDK_ROOT` | клон [NVIDIA/NVIDIAOpticalFlowSDK](https://github.com/NVIDIA/NVIDIAOpticalFlowSDK) (заголовки) | 3 |
| `TENSORRT_ROOT` | TensorRT 10.x (заголовки из OSS-репозитория NVIDIA/TensorRT + библиотеки из pip `tensorrt-cu12-libs` или zip с developer.nvidia.com) | 2 |
| `QT_ROOT` / `CMAKE_PREFIX_PATH` | Qt 6.8 msvc2022_64 | 4 |

Пример раскладки на машине разработки: `D:\SDK\DLSS`, `D:\SDK\Streamline`, `D:\SDK\OpticalFlowSDK`,
`D:\SDK\Qt\6.8.3\msvc2022_64`, `D:\SDK\dlssnr-patcher`.

## DLL в `bin/nvidia/`

| Файл | Откуда | Для чего |
|---|---|---|
| `nvngx_dlss.dll` | DLSS SDK `lib/Windows_x86_64/rel/` | DLSS Super Resolution (этап 5) |
| `nvngx_dlssg.dll`, `sl.*.dll` | Streamline | Frame Generation (этап 7) |
| `nvngx_dlssnr.dll` | из драйвера/игры с DLSS 5 (официально только RTX 50) | Neural Rendering (этап 6) |
| `nvngx_dlssnr.dll` (пропатченная) | результат `dlssnr-patcher` над вашей копией | NR на RTX 20/30/40 |

Приложение ищет DLL в `bin/nvidia/` рядом с exe (переопределяется `DLSSVID_NVIDIA_DLL_DIR`). При
отсутствии DLL соответствующая стадия отключается с сообщением; приложение не падает.

## Neural Rendering на RTX 40 (и 20/30)

1. Драйвер ≥ **616.56**. Проверка при старте стадии NR.
2. Скопируйте оригинальную `nvngx_dlssnr.dll` в безопасное место.
3. Патч на вашей машине: `python D:\SDK\dlssnr-patcher\dlssnr_patcher.py --ada "C:\path\to\nvngx_dlssnr.dll"`
   (кнопка «Пропатчить DLL» в GUI делает то же). Патчеру нужны `ptxas`, `fatbinary`, `cuobjdump`
   из **CUDA Toolkit 13.3** (`--cuda-bin <путь к bin>`); это отдельная установка от CUDA 12.4, с которой собирается проект.
4. Результат кладётся в `bin/nvidia/nvngx_dlssnr.dll`. Стадия NR логирует SHA-256 загруженной DLL.
5. Патченная DLL не подписана: антивирус может её блокировать — добавьте исключение для `bin/nvidia/`.
   Не используйте её с античит-защищёнными программами.
6. Проверенные пары «SHA-256 DLL + версия драйвера» будут перечислены здесь по мере тестов (этап 6).

Готовые патченные DLL (HuggingFace, Discord) в продукт не включаются — только ориентир для тестов.
