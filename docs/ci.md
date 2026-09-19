# CI: GitLab на self-hosted Windows-раннере с RTX

Пайплайн — `.gitlab-ci.yml`: job `build-test` (configure → build → все тесты, включая GPU-тесты и golden) и job
`package` (CPack ZIP / NSIS, только для `main` и тегов). Оба выполняются скриптами из `scripts/`, которые можно
запускать и вручную из корня репозитория:

```bat
scripts\ci-build.cmd     :: то же, что делает раннер: cmake --preset release, сборка, ctest (+ test-report.xml)
scripts\package.cmd      :: сборка и cpack --preset release -> build\release\dlss-video-<версия>-win64.zip
```

## Раннер (действие пользователя)

Раннер должен стоять на Windows-машине с RTX-видеокартой и тем же окружением, что у разработчика
(`docs/dll-setup.md`): Visual Studio 2022+ с C++ и Windows SDK, CMake ≥ 3.28, Ninja, vcpkg, CUDA 12.4, TensorRT 10.16,
Qt 6.8, клоны SDK и переменные окружения `VCPKG_ROOT`, `CUDA_PATH_V12_4`, `TENSORRT_ROOT`, `NV_OPTICAL_FLOW_SDK_ROOT`,
`QT_ROOT`, `DLSS_SDK_ROOT`, `DLSSVID_PYTHON` (venv с torch/onnx для экспорта моделей), `HF_TOKEN` (необязательно).
Переменные задаются **системно** (раннер работает как служба) или в конфиге раннера (`[[runners]] environment = [...]`).

1. Установить [gitlab-runner](https://docs.gitlab.com/runner/install/windows.html) (`gitlab-runner.exe install`,
   `gitlab-runner.exe start`; служба должна работать от пользователя с доступом к GPU — не `LocalSystem`).
2. В проекте `ai/video-dlss` на git.krem.digital: Settings → CI/CD → Runners → New project runner, теги `windows`,
   `rtx`, получить токен регистрации.
3. `gitlab-runner.exe register --url https://git.krem.digital --token <токен> --executor shell --shell pwsh
   --tag-list windows,rtx --description "rtx-4070-ti-super"` (или `--shell powershell`).
4. На машине раннера положить в `DLSSVID_NVIDIA_DLL_DIR` (переменная окружения службы) `nvngx_dlss.dll`,
   `nvngx_dlssg.dll` (из DLSS SDK — их копирует и сборка) и свою `nvngx_dlssnr.dll` — иначе GPU-тесты NR будут
   SKIP, а не красными.
5. Первый прогон скачивает веса моделей и собирает TensorRT-движки (`models/cache/` остаётся в рабочем каталоге
   раннера между сборками — `GIT_CLEAN_FLAGS` по умолчанию не трогает игнорируемые файлы).

Пока раннер не зарегистрирован, состояние CI проверяется локальным запуском `scripts\ci-build.cmd` (те же шаги).

## Что считается зелёным

- `ctest --preset release` — все тесты (unit, integration, app, golden); тесты, требующие DLL/модели, при их отсутствии
  помечаются SKIP и не валят сборку;
- `test-report.xml` (JUnit) прикладывается к job — GitLab показывает тесты во вкладке Tests;
- `package` — артефакт `dlss-video-<версия>-win64.zip`.
