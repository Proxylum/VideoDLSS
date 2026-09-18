# План этапа 1 — пассы и конвенции

Дата: 2026-09-18. Ветка: `stage/01-passes`. Источник требований: ТЗ §5 (пассы, форматы, импорт/экспорт,
конвертация raw ↔ DLSS), приёмка — ТЗ §8 (этап 1) и §9.

## Цель
Пассы — самостоятельные артефакты: последовательность файлов на кадр + `manifest.json`; экспорт и импорт
симметричны (round-trip без потерь, EXR float32), формулы raw ↔ DLSS зафиксированы в `docs/conventions.md`
и покрыты юнит-тестами до подключения любой DLSS-стадии.

## Что делаем
1. **`core/passes/`**
   - `PassImage` — CPU-изображение пасса: ширина, высота, каналы (имена), тип (`u8`, `u16`, `f16`, `f32`),
     данные interleaved. `PassKind` — `color_source`, `depth_raw`, `depth_dlss`, `mv_raw`, `mv_dlss`,
     `mask`, `color_sr`, `color_nr`, `color_fg`, `result` с каноническими каналами и типами.
   - `Manifest` — `manifest.json` (schema_version 1): имя пасса, разрешение, fps, число кадров, диапазон,
     формат, колорспейс, конвенция (`raw`/`dlss`), модель и версия, хэш исходного видео (SHA-256),
     параметры стадии, `depth.units` (`meters`/`relative`), `depth.near/far`, `mv.direction`/`mv.y_up`.
   - Форматы файлов (`formats/`): **EXR** (OpenEXR, float32/half, каналы по именам, multi-layer),
     **PNG 16-бит** (libpng, u8/u16), **TIFF** (libtiff, u8/u16/f32), **NPZ** (собственный reader/writer
     `.npy` внутри zip: stored при записи, stored + deflate при чтении — совместим с `numpy.savez`),
     **бинарь** `.r32` / `.rg16f` (сырой little-endian без заголовка, размер из манифеста).
   - `PassSequence` — папка пасса: чтение/запись манифеста, имена файлов `<pass>_%06d.<ext>`, экспорт из
     источника кадров с прогрессом и диапазоном, импорт с проверкой разрешения и числа кадров (ошибка с
     перечнем недостающих кадров), импорт без манифеста с указанием конвенции вручную.
   - Пресеты экспорта: `nuke` (multi-layer EXR: `R,G,B` + `depth.Z` + `mv.u,mv.v` в одном файле на кадр),
     `comfyui` (PNG16 для цвета + NPZ для depth/mv), `rawdlss` (бинарь + манифест).
2. **`core/convert/`**
   - `DepthConvert`: линейная глубина (метры) ↔ reverse-Z в [0,1] с near/far; relative-глубина —
     нормализация по min/max с сохранением параметров в манифесте; обратимость в пределах float32.
   - `MvConvert`: forward flow t→t+1 (px, y-down) → backward MV t→t−1 в конвенции DLSS: инверсия направления
     сплэттингом вперёд с разрешением коллизий по глубине (ближний побеждает), заполнение дыр ближайшим
     соседом, масштаб под целевое разрешение, дилатация 1–2 px по границам глубины; знак и ось Y —
     параметры `MvConvention` (по умолчанию: вектор от текущего пикселя к его позиции в предыдущем кадре,
     пиксели целевого разрешения, y-down — как в DLSS Programming Guide).
   - `ColorConvert`: YUV420P (BT.601/709, limited/full) → RGB float для `color_source`; обратная сторона
     не нужна (цветовой результат стадий пойдёт из GPU-текстур с этапа 2).
3. **CLI:** `dlssvid export -i video --pass color_source [--range a-b] [--format exr|png16|tiff|npz|raw]
   [--preset nuke|comfyui|rawdlss] -o dir`, `dlssvid import -i dir [--convention raw|dlss] [--near --far]
   [--expect WxH --expect-frames N]`, `dlssvid convert --from depth_raw|mv_raw -i dir -o dir [--near --far]
   [--target WxH] [--dilate 1]`.
4. **Тесты:** round-trip каждого формата и типа данных; manifest JSON; sequence export→import (все
   кадры, диапазон, ошибка при недостающих кадрах/несовпадении разрешения); формулы depth (обратимость,
   монотонность, near/far границы); MV (константный сдвиг, окклюзия по глубине, масштаб, дилатация);
   multi-layer EXR; NPZ, совместимый с numpy (проверка через `python -c "numpy.load"` при наличии numpy);
   CLI export/import/convert как процесс.
5. **Документация:** `docs/conventions.md` — формулы, единицы, знаки, форматы файлов, манифест.

## Принятые решения
- Пассы на CPU в этапе 1 (`PassImage`): GPU-текстуры пассов появятся вместе с первой считающей стадией (этап 2).
- Хэш исходного видео — SHA-256 файла (через bcrypt Windows API, без новых зависимостей).
- NPZ пишется без сжатия (stored): проще и без потерь; чтение поддерживает deflate (`numpy.savez_compressed`).
- Мультислойный EXR — только экспорт-пресет; импорт multi-layer раскладывает по пассам по именам каналов.

## Приёмка
- `ctest`: round-trip экспорт→импорт без потерь для всех пассов (EXR float32), тесты raw↔DLSS на синтетике.
- `docs/conventions.md` заполнен; CLI export/import/convert работают на клипе из этапа 0.

## Итог (2026-09-18)
- Реализовано всё из плана; `ctest --preset release`: 69/69 (44 новых теста этапа 1: форматы, манифест, half,
  depth/MV/цвет, последовательности, CLI как процесс, NPZ ↔ numpy).
- Инцидент сборки: `near`/`far` — макросы `<windows.h>`; поля и параметры переименованы в `zNear`/`zFar`
  (ключи манифеста остались `near`/`far`).
- Smoke на 1280x720 H.264: `export` (EXR half, 10 кадров, SHA-256 источника в манифесте), `--preset comfyui`,
  `import --expect-size --expect-frames` — ок.
- Отложено: GPU-текстуры пассов и дисковый кэш как часть `GpuFrameCache` (этап 2), импорт multi-layer EXR
  обратно в отдельные пассы через CLI (сейчас — `ReadExrLayers` в API), y_up-переворот при импорте (флаг в манифесте есть, применяется стадиями).
