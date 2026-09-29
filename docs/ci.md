# CI: GitHub Actions на self-hosted Windows-раннере с RTX

Два workflow. `.github/workflows/ci.yml`: job `build-test` (configure → build → все тесты, включая GPU-тесты и golden) на
каждый push в `main`, pull request и вручную (`workflow_dispatch`); job `package` (CPack ZIP / NSIS) — на push в `main`,
ZIP как артефакт. `.github/workflows/release.yml`: на теги `v*` — только `package`, пакет прикладывается к GitHub Release
тега (build-test не повторяется: тегируется merge-коммит, который уже прошёл тесты в pull request и на `main`; на одном
раннере это экономит 20+ минут на релиз). Все job выполняются скриптами из `scripts/`, которые можно запускать и вручную
из корня репозитория:

```bat
scripts\ci-build.cmd     :: то же, что делает раннер: configure, сборка, ctest (+ test-report.xml)
scripts\package.cmd      :: сборка и cpack --preset release -> build\release\dlss-video-<версия>-win64.zip
scripts\configure.cmd    :: cmake --preset release с повтором до 3 раз (vcpkg install тянет реестр с GitHub, и с раннера
                         :: это время от времени срывается — «Fetching registry information … failed»; повтор проходит)
```

Раннер — машина с NVIDIA RTX (labels `self-hosted`, `windows`, `rtx`): GPU-тесты, DLSS SR / NR / FG, TensorRT-движки идут
на настоящем железе; hosted-раннеры GitHub для этого не подходят.

## Раннер

Официальный [actions/runner](https://github.com/actions/runner) (ZIP с страницы релиза, sha256 сверяется с опубликованной)
распакован в `D:\actions-runner` (пример; любой короткий путь) и зарегистрирован на репозиторий как self-hosted runner с
labels `windows`, `rtx`. Работает как **процесс в сессии пользователя**, не как служба: так у него есть GPU и окружение
пользователя (бинарный кэш vcpkg, venv). Всё, что нужно сборке, задано в файле `.env` в папке раннера (раннер читает его
при старте и отдаёт job'ам):

| Переменная | Значение | Зачем |
|---|---|---|
| `VSLANG` | `1033` | английские сообщения MSVC → ninja видит зависимости от заголовков (инцидент этапа 8) |
| `VCPKG_ROOT`, `TENSORRT_ROOT`, `NV_OPTICAL_FLOW_SDK_ROOT`, `QT_ROOT`, `DLSS_SDK_ROOT`, `DLSSNR_PATCHER_ROOT`, `VDA_REPO`, `SEARAFT_REPO` | `C:\vcpkg`, `<SDK>\...` — то, что пишет `scripts\setup_sdk.py` в `<SDK>\env.cmd` | configure и экспорт моделей (`docs/dll-setup.md`) |
| `DLSSVID_NVIDIA_DLL_DIR` | `...\dlss-video\build\release\bin\nvidia` дерева разработчика | `nvngx_dlss.dll`, `nvngx_dlssg.dll`, `nvngx_dlssnr.dll` — иначе GPU-тесты SR/FG/NR были бы SKIP |
| `DLSSVID_MODELS_DIR` | `...\dlss-video\models` дерева разработчика | `registry.json` + `cache/` с весами, ONNX и TensorRT-движками — CI не качает модели и не собирает движки заново |
| `DLSSVID_PYTHON` | `...\models\export\.venv\Scripts\python.exe` | воркеры/экспорт, если тесту всё же понадобится Python |
| `DLSSVID_TENSORRT_DIR` | `...\models\export\.venv\Lib\site-packages\tensorrt_libs` | рантайм TensorRT (`nvinfer_10.dll` и др.): в `TENSORRT_ROOT` только заголовки, DLL ставит `pip install tensorrt-cu12` в venv |

Git на машине раннера должен уметь длинные пути (`git config --global core.longPaths true`): деревья сборки vcpkg длиннее
MAX_PATH, и без этого `git clean -ffdx`, которым `actions/checkout` (`clean: true`) чистит рабочую папку перед job, падает.
Рабочая папка — `D:\actions-runner\_work\VideoDLSS\VideoDLSS`: клон с `fetch-depth: 50`, перед каждым job чистка удаляет и
игнорируемые файлы, то есть `build/` — каждый job собирает проект с нуля (vcpkg восстанавливает пакеты из бинарного кэша
пользователя). Это намеренно: чистая сборка ловит устаревшие объекты и пропущенные зависимости.

Тесты запускаются из папки **вне** клона — `%TEMP%\dlssvid-test-cwd` (`DLSSVID_TEST_CWD` в `tests/CMakeLists.txt`,
`WORKING_DIRECTORY` у всех `catch_discover_tests`). Причина: каждая инициализация NGX (DLSS / NR / FG) в тестах порождает
обновлятор NVIDIA `nvngx_update.exe -api update -feature …` (OTA), он наследует рабочую папку тестового процесса и живёт
ещё до минуты после ctest; с рабочей папкой внутри клона следующий job не мог удалить её при чистке («Permission denied»).
Тесты работают только с абсолютными путями (`DLSSVID_TEST_TMP`, `DLSSVID_CLI_PATH`, `DLSSVID_GOLDEN_DIR`), поэтому папка не
важна. Отключить OTA можно только машинно (реестр `HKLM\SOFTWARE\NVIDIA Corporation\Global\NGXCore\EnableOTA`), в
приложении такого переключателя у NGX нет.

### Как он запущен и как перезапустить

Служба Windows требует прав администратора (`svc.cmd install`), поэтому раннер работает как обычный процесс в сессии
пользователя:

```bat
D:\actions-runner\run.cmd
```

Автозапуск — задача планировщика `GitHub Actions Runner (VideoDLSS)` (при входе пользователя, без ограничения времени
выполнения, одна копия): она запускает `D:\actions-runner\start-runner.ps1`, который стартует `run.cmd` скрытым процессом
(`Start-Process -WindowStyle Hidden`, логи — `D:\actions-runner\runner.out.log` / `runner.err.log`) и не запускает вторую
копию, если раннер уже работает (`Runner.Listener.exe` из этой папки). Тот же скрипт можно запустить вручную:
`powershell -ExecutionPolicy Bypass -File D:\actions-runner\start-runner.ps1`.

Проверка: в репозитории Settings → Actions → Runners (статус Idle / Active) или `gh api repos/<владелец>/<репозиторий>/actions/runners`.

### Как зарегистрировать заново (другая машина, отозванная регистрация)

1. Скачать ZIP раннера со страницы релиза `actions/runner`, сверить sha256 с опубликованным, распаковать в короткий путь.
2. Токен регистрации: Settings → Actions → Runners → New self-hosted runner (показывает команду с токеном), или через API
   с правами администратора репозитория: `gh api -X POST repos/<владелец>/<репозиторий>/actions/runners/registration-token`.
   Токен живёт час и нигде не сохраняется.
3. `config.cmd --unattended --url https://github.com/<владелец>/<репозиторий> --token <токен> --name <имя> --labels windows,rtx --work _work`.
4. Положить `.env` из таблицы выше (пути под свою машину; `scripts\setup_sdk.py --env` печатает SDK-часть) и запустить
   `run.cmd` (или зарегистрировать задачу планировщика, как выше).

Снять раннер: `config.cmd remove --token <токен удаления>` (`gh api -X POST …/actions/runners/remove-token`), удалить задачу
планировщика и папку.

## Что считается зелёным

- `ctest --preset release` — все тесты (unit, integration, app, golden); тесты, требующие DLL/модели, при их отсутствии
  помечаются SKIP и не валят сборку;
- `test-report.xml` (JUnit) и `LastTest.log` прикладываются к job как артефакт `test-report-<номер запуска>`;
- `package` — артефакт `dlss-video-package-<ref>` с `dlss-video-<версия>-win64.zip`; на теге (release.yml) тот же ZIP в Release.
