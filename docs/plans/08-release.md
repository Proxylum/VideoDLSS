# План этапа 8 — сборка, CLI, документация, релиз

Дата: 2026-09-19. Ветка: `stage/08-release`. Требования: ТЗ §8 (этап 8: инсталлер, `dll-setup.md`, бенчмарк-команда,
golden-тесты, CI; приёмка — «чистая установка на машине с RTX проходит все golden-тесты»), §9 («тестовый клип 1080p проходит
весь пайплайн до 4K с FG ×2 из CLI и из GUI, все пассы экспортируются и импортируются обратно»; «все возможности доступны
из CLI; `dlssvid bench` печатает время на кадр по стадиям»), §7 (`cli/`: `process`, `export`, `import`, `bench`;
«CI на self-hosted Windows-раннере с RTX» — у нас GitLab CI на своём раннере), §4 («дисковый кэш пассов… чтобы повторный
прогон с другими параметрами NR/FG не пересчитывал предыдущие стадии»), §5 (`result`: MP4/MOV + копия аудио).

## Разведка (2026-09-19)

| Что | Статус | Решение |
|---|---|---|
| `dlssvid process` | только `--passthrough` (этап 0): `RunPipeline` со стадией passthrough, аудио копируется пакетами | полный пайплайн как **последовательность прогонов стадий по дисковому кэшу пассов** (`RunDepth` → `RunFlow` → `RunUpscale` → `RunNr` → `RunFg`, каждая читает пассы предыдущих из `<passes>/`), затем **encode** финального цветового пасса (`color_fg` › `color_nr` › `color_sr`) с копией аудио из источника. Стадии, чей пасс уже полный, пропускаются (`--skip-existing`, по умолчанию) — это и есть «повторный прогон не пересчитывает». Параметры стадий — те же JSON-ключи, что в файле проекта (`Project::stages`) и в GUI: `RunUpscale/RunNr/RunFg` получают `params` через `StageConfig` (их `ApplyParams`), depth/flow — маппинг в раннере |
| GUI | стадии запускаются по одной («Запустить»); «result» как источник вьюпорта поддержан (`Project::resultVideo`, `Project::Sources()`), play для `color_fg` уже с удвоенным fps (`AppModel::setPlaying`) | кнопка **«Обработать → result»** в панели проекта: сохранить проект, запустить `dlssvid process --project <файл>` в `TaskQueue`, после завершения `reloadSources()` показывает `result` |
| `dlssvid bench` | нет | `RunProcess` во временный корень пассов без пропусков + отчёт: GPU / архитектура / драйвер, мс/кадр каждой стадии и encode, JSON (`--json`); результаты в `docs/benchmarks.md` |
| Golden-тесты | `tests/golden/` пуст; интеграционные тесты покрывают стадии по отдельности | отдельный исполняемый `dlssvid_golden_tests` (ctest label `golden`): (1) детерминированный сквозной прогон на WARP — синтетический клип 96×54 с аудио, стадии `depth stub → flow stub → upscale nis ×2 → nr stub → fg blend ×2` → сравнение кадров `color_sr`/`color_nr`/`color_fg` с эталонами `tests/golden/ref/*.png` (PSNR ≥ 45 дБ; `DLSSVID_GOLDEN_UPDATE=1` перезаписывает эталоны), число кадров и fps пассов и результата, копия аудио, round-trip экспорт → импорт `depth_raw`/`mv_raw` (EXR float32) без потерь; (2) `[gpu]` — реальный пайплайн (`da3`, `ofa`, `dlss`, `ngx`, `dlssg`) на том же клипе: счётчики, диапазоны, PSNR к соседям, SKIP без DLL/GPU. `DLSSVID_CLI` переопределяет путь к `dlssvid.exe` → тот же тест проверяет **установленную** копию |
| Инсталлер | инструментов NSIS/Inno/WiX на машине нет; всё рантайм-окружение уже лежит в `build/<preset>/bin/` (vcpkg-DLL, Qt через windeployqt, `nvidia/`) | `install()` + **CPack ZIP** (переносимая раскладка `bin/`, `models/registry.json` + `models/export/`, `depth_worker/`, `docs/`, README) — без `*.pdb`/`*.ilk`/тестов и **без `bin/nvidia/*.dll`** (NVIDIA не распространяем: там только README); NSIS-инсталлер генерируется тем же CPack, когда установлен `makensis` (`docs/release.md`); package-preset в `CMakePresets.json`; `scripts/package.cmd` |
| CI | `.gitlab-ci.yml` нет; `gitlab-runner` на машине не установлен; регистрация раннера требует токен проекта (действие пользователя) | `.gitlab-ci.yml` (job `build-test` на тегах `windows`, `rtx`: `scripts/ci-build.cmd` → vswhere/vcvars64 → `cmake --preset release` → build → ctest (включая GPU) → golden; job `package` → `cpack`, артефакт ZIP) + `docs/ci.md` (установка и регистрация раннера, переменные окружения); скрипт проверяется локально теми же шагами; «CI зелёный» ждёт регистрации раннера (TASK-0011) |
| Документация | `dll-setup.md` актуален по SR/NR/FG | финальный проход: README (установка из ZIP, полный прогон CLI/GUI, `bench`), `docs/release.md` (сборка пакета, что внутри, что пользователь докладывает сам), `docs/architecture.md` (ProcessRunner), `docs/benchmarks.md` (bench) |

## Что делаем

1. **`core/pipeline/ProcessRunner`**: `ProcessOptions{input, output, passesRoot, stages[{name, enabled, params}], skipExisting,
   codec/codecOptions, frames, warp, hwaccel, disableUnavailable, progress}` → `RunProcess()`: канонический порядок
   `depth, flow, upscale, nr, fg`, пропуск полных пассов, отчёт по стадиям (ran / reused / disabled, мс/кадр, кадры),
   `EncodePassToVideo` (кадры пасса с его fps ↔ кадры источника по `mult = round(passFps / sourceFps)`, аудио пакетами
   через `SetAudioPacketSink`, чересстрочно с видео), без цветовых стадий — passthrough как раньше. `RunUpscale/RunNr/RunFg`
   получают `params`.
2. **CLI** `cli/ProcessCommands`: `dlssvid process -i in.mp4 -o out.mp4 [--passes dir] [--project p.dlssvid.json] [--stages
   depth,flow,upscale,nr,fg] [--<stage>.<key> value…]` — параметры стадий через `--param depth.backend=da3` (повторяемый) и
   короткие алиасы (`--scale`, `--upscale-backend`, `--nr-backend`, `--fg-backend`, `--multiplier`), `--no-skip-existing`,
   `--disable-unavailable`, `--codec`, `--frames`, `--warp`, `--passthrough` (старое поведение), `--json`; `dlssvid bench -i clip
   [--frames 30] [--stages …] [--json]`.
3. **GUI**: кнопка «Обработать → result» (`ProjectPanel`), результат — `<источник>_result.mp4` (или `Project::resultVideo`),
   после задачи `reloadSources()`; `docs/app` README.
4. **Golden**: `tests/golden/{test_golden.cpp, expected.json, ref/}` + цель `dlssvid_golden_tests` (label `golden`),
   `DLSSVID_CLI` для установленной копии; `tests/CMakeLists`.
5. **Пакет**: `install()`-правила, CPack (ZIP; NSIS при `makensis`), `packagePresets`, `scripts/package.cmd`, `bin/nvidia/README.md`.
6. **CI**: `.gitlab-ci.yml`, `scripts/ci-build.cmd`, `docs/ci.md`.
7. **Документация и память**: README, `docs/release.md`, `docs/architecture.md`, `docs/benchmarks.md`, `docs/dll-setup.md`
   (финальный проход), `Memory/*`, TASK-0011 (+ регистрация раннера).

## Открытые вопросы
- Регистрация GitLab-раннера (токен проекта, установка `gitlab-runner` как службы) — действие пользователя; до неё CI
  проверяется локальным запуском `scripts/ci-build.cmd`.
- 4K-прогон 1080p-клипа (приёмка §9) требует реальный клип (TASK-0010); на smoke-клипе 720p → 1440p ×2 FG проверяется
  тем же `process`.

## Отложено / вне этапа
- NSIS-инсталлер с ярлыками (CPack готов, нужен `makensis` на машине сборки); подпись бинарников; автообновление.

## Приёмка (ТЗ §8–9)
- `dlssvid process` и кнопка GUI дают `result` (SR → NR → FG ×2 + аудио) на smoke-клипе; повторный прогон переиспользует
  пассы; `dlssvid bench` печатает мс/кадр по стадиям; golden-тесты проходят из build и из распакованного ZIP;
  `.gitlab-ci.yml` + скрипт проходят локально; документация обновлена; PR.

## Итог (2026-09-19)

- `core/pipeline/ProcessRunner` (`RunProcess`: стадии над кэшем пассов, переиспользование полных пассов — в т. ч. для
  контейнеров без счётчика кадров по хэшу источника, `EncodePassToVideo` с аудио и темпом по кадрам источника,
  passthrough), `RunUpscale/RunNr/RunFg` принимают JSON-параметры стадий; CLI `dlssvid process` (`--project`, `--stages`,
  `--param stage.key=value`, алиасы, `--no-skip-existing`, `--disable-unavailable`, `--json`) и `dlssvid bench`; кнопка
  «Обработать → result» в GUI; golden-тесты (`dlssvid_golden_tests`: детерминированный пайплайн на WARP против эталонов
  `tests/golden/ref/*` в EXR + реальные бэкенды `[gpu]`, `DLSSVID_CLI`/`DLSSVID_GOLDEN_DIR` для установленной копии);
  `install()` + CPack ZIP (NSIS при `makensis`), `packagePresets`, `scripts/package.cmd`; `.gitlab-ci.yml` +
  `scripts/ci-build.cmd` + `docs/ci.md`; `docs/release.md`.
- Проверено: полный пайплайн на smoke-клипе 720p → `result.mp4` 2560×1440 @ 60 fps (SR ×2 + NR + FG ×2, AAC скопирован) за
  48.6 с / 30 кадров; `bench` — таблица в `docs/benchmarks.md`; пакет `dlss-video-0.1.0-win64.zip` (30 МБ, 69 МБ
  распакованный, без `*.pdb`, тестов и DLL NVIDIA) собран и golden-тест пройден **на распакованной копии**;
  `scripts/ci-build.cmd` проходит локально (configure → build → 179 тестов, JUnit-отчёт).
- Инцидент сборки: ninja не записывал зависимости от заголовков у части объектов (локализованный префикс `/showIncludes`
  MSVC) — истинная причина «устаревших объектов» этапа 4; регрессионный тест `build.ninja_header_deps`, `VSLANG=1033` в
  пресетах, чистая пересборка; корневое исправление — английский языковой пакет VS (TASK-0011).
- Тесты: +`test_process`, `test_cli_process`, golden ×2, `build.ninja_header_deps`; всего 179.
- Не проверено: «CI-пайплайн зелёный» в GitLab — раннер не зарегистрирован (действие пользователя, TASK-0011);
  4K-прогон реального 1080p-клипа — после TASK-0010.
- Отложено: NSIS-инсталлер (нужен `makensis`), демукс аудио без декодирования видео, потоковый пайплайн без CPU-readback,
  воспроизведение `color_fg` во вьюпорте с fps × mult (уже удваивается для `color_fg` как базового слоя).
