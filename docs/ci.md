# CI: GitLab на self-hosted Windows-раннере с RTX

Пайплайн — `.gitlab-ci.yml`: job `build-test` (configure → build → все тесты, включая GPU-тесты и golden) и job
`package` (CPack ZIP / NSIS, только для `main` и тегов). Оба выполняются скриптами из `scripts/`, которые можно
запускать и вручную из корня репозитория:

```bat
scripts\ci-build.cmd     :: то же, что делает раннер: cmake --preset release, сборка, ctest (+ test-report.xml)
scripts\package.cmd      :: сборка и cpack --preset release -> build\release\dlss-video-<версия>-win64.zip
```

## Раннер

Раннер стоит на машине с RTX (в примерах ниже — `D:\GitLab-Runner`, RTX 4070 Ti SUPER) и зарегистрирован в проекте
как project runner с тегами `windows`, `rtx` (shell executor, Windows PowerShell 5.1). Всё, что ему нужно, задано
в `D:\GitLab-Runner\config.toml` (`[[runners]] environment = [...]`), потому что в системном окружении этих
переменных нет:

| Переменная | Значение | Зачем |
|---|---|---|
| `VSLANG` | `1033` | английские сообщения MSVC → ninja видит зависимости от заголовков (инцидент этапа 8) |
| `VCPKG_ROOT`, `TENSORRT_ROOT`, `NV_OPTICAL_FLOW_SDK_ROOT`, `QT_ROOT`, `DLSS_SDK_ROOT` | `C:\vcpkg`, `D:\SDK\...` | configure (`docs/dll-setup.md`) |
| `DLSSVID_NVIDIA_DLL_DIR` | `...\dlss-video\build\release\bin\nvidia` дерева разработчика | `nvngx_dlss.dll`, `nvngx_dlssg.dll`, `nvngx_dlssnr.dll` — иначе GPU-тесты SR/FG/NR были бы SKIP |
| `DLSSVID_MODELS_DIR` | `...\dlss-video\models` дерева разработчика | `registry.json` + `cache/` с весами, ONNX и TensorRT-движками — CI не качает модели и не собирает движки заново |
| `DLSSVID_PYTHON` | `...\models\export\.venv\Scripts\python.exe` | воркеры/экспорт, если тесту всё же понадобится Python |
| `DLSSVID_TENSORRT_DIR` | `...\models\export\.venv\Lib\site-packages\tensorrt_libs` | рантайм TensorRT (`nvinfer_10.dll` и др.): в `TENSORRT_ROOT` только заголовки, DLL ставит `pip install tensorrt-cu12` в venv — без этой переменной первый прогон упал на golden-тесте с реальными бэкендами («backend 'da3' needs models/export/.venv») |

Git на машине раннера должен уметь длинные пути (`git config --global core.longPaths true` — сделано; `.gitlab-ci.yml`
дополнительно задаёт то же через `GIT_CONFIG_KEY_0/VALUE_0`): деревья сборки vcpkg длиннее MAX_PATH, и без этого
`git clean` между job падает. Каталог сборки — `D:\GitLab-Runner\builds\<runner>\0\<группа>\<проект>`: клон с `GIT_DEPTH=50`, перед каждым job
`git clean -ffdx` (значение `GIT_CLEAN_FLAGS` по умолчанию) удаляет и игнорируемые файлы, то есть `build/` —
каждый job собирает проект с нуля (vcpkg восстанавливает пакеты из бинарного кэша пользователя). Это намеренно:
чистая сборка ловит устаревшие объекты и пропущенные зависимости.

Тесты запускаются из папки **вне** клона — `%TEMP%\dlssvid-test-cwd` (`DLSSVID_TEST_CWD` в `tests/CMakeLists.txt`,
`WORKING_DIRECTORY` у всех `catch_discover_tests`). Причина: каждая инициализация NGX (DLSS / NR / FG) в тестах порождает
обновлятор NVIDIA `nvngx_update.exe -api update -feature …` (OTA), он наследует рабочую папку тестового процесса и живёт
ещё до минуты после ctest. С рабочей папкой по умолчанию (`build/release/tests`) следующий job (`package` на `main`)
не мог удалить её при `git clean` — «failed to remove build/release/tests: Permission denied», job падал ещё до
скрипта (три пайплайна подряд 2026-09-22; тот же симптом дважды вручную чинился во время MR D). Тесты
работают только с абсолютными путями (`DLSSVID_TEST_TMP`, `DLSSVID_CLI_PATH`, `DLSSVID_GOLDEN_DIR`), поэтому папка не
важна. Отключить OTA можно только машинно (реестр `HKLM\SOFTWARE\NVIDIA Corporation\Global\NGXCore\EnableOTA`), в
приложении такого переключателя у NGX нет.

### Как он запущен и как перезапустить

Служба Windows требует прав администратора (`gitlab-runner install`), поэтому раннер работает как обычный процесс
в сессии пользователя — так у него есть и GPU, и окружение пользователя (бинарный кэш vcpkg):

```bat
D:\GitLab-Runner\gitlab-runner.exe run --config D:\GitLab-Runner\config.toml --working-directory D:\GitLab-Runner
```

Автозапуск — задача планировщика `GitLab Runner (video-dlss)` (при входе пользователя, без ограничения времени
выполнения, одна копия): она запускает `D:\GitLab-Runner\start-runner.ps1`, который стартует раннер скрытым процессом
(`Start-Process -WindowStyle Hidden`, логи — `D:\GitLab-Runner\runner.err.log`) и не запускает вторую копию, если
раннер уже работает. Тот же скрипт можно запустить вручную:
`powershell -ExecutionPolicy Bypass -File D:\GitLab-Runner\start-runner.ps1`. Служба (`gitlab-runner install`) не
использована: нужны права администратора, а служба от `LocalSystem` не увидела бы кэш vcpkg и venv пользователя.

Полное удаление раннера с машины: завершить процесс `gitlab-runner.exe`, удалить задачу планировщика,
`D:\GitLab-Runner\gitlab-runner.exe unregister --config D:\GitLab-Runner\config.toml --all-runners` (снимает
регистрацию в GitLab), удалить папку `D:\GitLab-Runner`.

Проверка: `D:\GitLab-Runner\gitlab-runner.exe verify --config D:\GitLab-Runner\config.toml`; в GitLab —
Settings → CI/CD → Runners (зелёная точка «online»).

### Как зарегистрировать заново (другая машина, отозванный токен)

1. Токен раннера: Settings → CI/CD → Runners → New project runner (теги `windows`, `rtx`) — или через API с
   personal access token со scope `api`: `POST /api/v4/user/runners` с `runner_type=project_type`,
   `project_id=<id проекта>`, `tag_list=windows,rtx`, `locked=true` (ответ содержит `token`).
2. `gitlab-runner.exe register --non-interactive --url https://<хост GitLab> --token <glrt-…> --executor shell
   --shell powershell --builds-dir D:\GitLab-Runner\builds --cache-dir D:\GitLab-Runner\cache --config D:\GitLab-Runner\config.toml`.
3. Добавить в `config.toml` строку `environment = [...]` из таблицы выше (пути под свою машину) и запустить.

Пока раннер не работает, состояние CI проверяется локальным запуском `scripts\ci-build.cmd` (те же шаги).

## Что считается зелёным

- `ctest --preset release` — все тесты (unit, integration, app, golden); тесты, требующие DLL/модели, при их отсутствии
  помечаются SKIP и не валят сборку;
- `test-report.xml` (JUnit) прикладывается к job — GitLab показывает тесты во вкладке Tests;
- `package` — артефакт `dlss-video-<версия>-win64.zip`.
