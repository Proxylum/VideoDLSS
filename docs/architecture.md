# Архитектура

Полная целевая архитектура — ТЗ §4 (граф стадий, кэш кадров, контракты). Здесь — то, что
реализовано, и хаки, которые нужно помнить.

## Состояние после этапов 0–8

```
VideoDecoder ──CpuFrame(YUV420P)──▶ Pipeline ──▶ VideoEncoder
   FFmpeg                              │            FFmpeg + NVENC / ffv1
   (sw | NVDEC→CPU)                    ▼            audio: stream copy
                              [ PassthroughStage ]
                                       │ upload / readback
                                       ▼
                              GpuFrameCache (ring of D3D12 buffers)
                                       │
                              D3D12Device (single device + direct queue)
                                       │ shared heap → external memory
                              CudaInterop (cudaImportExternalMemory)
```

- `IStage` — `Init(StageConfig, D3D12Device)`, `Process(FrameContext)`, `Shutdown()`; `StageConfig.params` — JSON.
- `Pipeline` — линейный список; граф/планировщик появится, когда стадий станет больше одной (этап 2).
- `GpuFrameCache` — кольцо из N слотов; сейчас в слоте один буфер с байтами кадра. Стадии 1–3 добавят
  в слот текстуры пассов (`color` RGBA16F, `depth_raw` R32F, `mv_raw` RG16F …).
- `CudaInterop` — выбор CUDA-девайса по LUID адаптера D3D12; импорт буфера, созданного на
  `D3D12_HEAP_FLAG_SHARED`. Синхронизация пока блокирующая (`cudaDeviceSynchronize` + fence);
  внешние семафоры (`cudaImportExternalSemaphore` ↔ `ID3D12Fence`) — этап 2.

### Слой пассов (этап 1)

```
PassImage (CPU, interleaved, u8/u16/f16/f32)        core/passes/PassImage.h
   │  PassKind: color_source, depth_raw, depth_dlss, mv_raw, mv_dlss, mask, color_sr/nr/fg, result
   ▼
PassFile  ──▶ formats/ExrIO · PngIO · TiffIO · NpzIO · RawIO   (диспетчер по расширению)
   ▼
PassWriter / PassReader  (папка = файлы <pass>_%06d.<ext> + manifest.json)   core/passes/PassSequence.h
   ▼
ExportPass / ExportLayeredExr / presets (nuke, comfyui, rawdlss)
convert/: DepthConvert (raw ↔ reverse-Z), MvConvert (InvertFlow, dilation, scale), ColorConvert (YUV → RGB)
```

- Пассы живут на CPU (`PassImage`) и на диске; GPU-текстуры пассов добавляются в слот
  `GpuFrameCache` вместе с первой считающей стадией (этап 2). Дисковый кэш пассов (ТЗ §4) — это
  те же папки `PassWriter`/`PassReader` в каталоге проекта.
- Все формулы — `docs/conventions.md`; они реализованы один раз в `core/convert/` и не дублируются в стадиях.
- `Manifest` — единственный источник геометрии для бинарных дампов и канонических имён каналов;
  `PassReader::Validate` формулирует несовпадения (разрешение, число кадров, пропуски) одним сообщением.

### Стадия Depth (этап 2)

```
CpuFrame(YUV) ─ColorConvert─▶ RGB F32 ─▶ DepthStage (окно N кадров, overlap) ─▶ IDepthEstimator
                                                                                   ├─ TrtDepthEstimator (da3, vda): resize→NCHW→TensorRT FP16→disparity/depth→guided upsample
                                                                                   ├─ WorkerDepthEstimator (worker:da3|vda|icdepth|stub): python depth_worker, .npz через scratch
                                                                                   └─ StubDepthEstimator (тесты)
                                                            ▼
                             FillInvalidDepth → TemporalStabilizer (scale/shift, окно 8) → TAE → PassWriter(depth_raw, depth_dlss) + GpuFrameCache slot texture R32F
```

- TensorRT грузится в рантайме (`ml/TrtLoader`): C-точки входа заголовков (`createInferRuntime_INTERNAL` …)
  определены у нас и форвардят в `nvinfer_10.dll`; import-библиотеки не нужны, без DLL бэкенды `da3`/`vda`
  выдают понятную ошибку, остальное работает.
- ONNX под конкретную геометрию (`<model>_<T>x<H>x<W>.onnx`) экспортируется при первом обращении скриптом
  из venv; engine кэшируется по хэшу ONNX + GPU + версия TensorRT + fp16.
- `IStage::Finish()` добавлен для оконных стадий (сброс хвоста окна после последнего кадра).
- Метрика TAE без MV — прокси (|d_t − d_{t−1}| / mean d); warp-версия — этап 3.

### Стадия Motion vectors (этап 3)

```
NVDEC (CUDA NV12, primary context) ──GpuFrame──▶ FlowStage ──▶ IFlowEstimator
   │ (CPU-копия только по запросу)                 │  задержка в 1 кадр    ├─ OfaFlowEstimator: cudaMemcpy2D NV12 → буферы nvofapi → S10.5 → float
   └──CudaInterop::CopyNv12──▶ GpuFrameCache slot  │                      ├─ TrtFlowEstimator (searaft): resize → 2×NCHW → TensorRT → ScaleMv
      (shared heap, NV12 layout, без CPU)          ▼                      └─ StubFlowEstimator
                                     mv_raw[t] (forward) ─ForwardFlowToBackwardMv(+depth)─▶ mv_dlss[t+1] ─▶ PassWriter + GpuFrameCache textures
                                                                            └─ WarpPsnr(prev, cur, mv_dlss) → статистика/манифест
```

- `FrameContext.gpu` — NVDEC-кадр на устройстве; `Pipeline::ProcessFrame(frame, gpu)`; `RunPipeline`/`RunFlow`
  передают его при `--hwaccel cuda`. `GpuFrameCache::AttachCuda` создаёт слоты на shared-heap и импортирует их в
  CUDA, `UploadFromGpuFrame` копирует NV12 device-to-device (readback распаковывает в YUV420P).
- OFA держит копию предыдущего NV12-кадра в собственной device-памяти (декодер переиспользует свой буфер).
- Глубина для окклюзий в `mv_dlss` берётся из папки `depth_raw` (`--depth-dir`) или из слота кэша, если DepthStage
  отработала в том же пайплайне.


### Вьюпорт (этап 4)

```
FrameStore (GPU-кэш кадров)                    ViewportRenderer (один composite-шейдер)
  ├─ loader-потоки (2–4): пассы через PassReader,   ├─ Composite.hlsl: до 5 слоёв, colormaps, MV HSV/magnitude,
  │  видео через VideoDecoder (один декодер на       │  маски fill/contour, blend normal/difference/multiply/screen,
  │  источник, seek к ключевому кадру + проход        │  wipe, checkerboard вне кадра; 2x2 = 4 draw с viewport/scissor
  │  вперёд, кадры «по пути» из окна префетча        ├─ Arrows.hlsl: инстансированные линии, VS читает MV-текстуру
  │  сохраняются) → RGBA16F / R32F / RG32F / R8      ├─ swapchain (QWindow) или offscreen RGBA8 (PNG, CLI, тесты)
  ├─ Update() на render-потоке: upload ≤ N/кадр,      └─ ReadTexel: 1×1 readback для пробника
  │  LRU-вытеснение по бюджету VRAM вне окна ±prefetch
  └─ статусы Missing / Queued / Loading / Ready; время → кадр каждого источника (FrameAt, TexturesAt)
ViewportState (JSON в проекте) ── Project (*.dlssvid.json: source, passes, result, stages, viewport)
AppModel (Qt) ── ViewportWindow / панели ── TaskQueue (dlssvid <stage> как процесс, прогресс из "N/M frames")
```

- Один и тот же путь для GUI и CLI: `dlssvid render` собирает `ViewportState` из проекта и опций, ждёт
  `FrameStore::WaitForCurrent()` и рендерит offscreen; GUI рендерит в swapchain по таймеру 16 мс и
  подхватывает кадры по мере готовности (`FrameStore::Update()`).
- Видео в кэше — CPU-декод (FFmpeg, frame/slice-потоки) → `Yuv420pToRgba16f` (F16C) → upload. На 1080p
  (RTX 4070 Ti SUPER, Ryzen 7 7700X): холодный скраббинг одного источника 12 мс/кадр (82 fps), сетка 2x2
  из четырёх 1080p-источников 27 мс/кадр (37 fps), тёплый (кадры в кэше) < 1 мс, случайные прыжки по
  H.264 с GOP 250 — 40 мс (seek + декод от ключевого кадра); composite сам по себе 0.2–2.5 мс.
- Проект хранит относительные пути (переносим вместе с папкой), состояние вьюпорта и конфиги стадий;
  панель проекта запускает `dlssvid depth|flow` с параметрами из JSON стадии.
- **Временная ось (этап 9, MR B).** Позиция вьюпорта — время в секундах (`ViewportState::time`; поле `frame` старых
  проектов читается как `legacyFrame` и переводится во время по базовой частоте, когда источники известны). У каждого
  источника своя частота: кадр = round(t × fps источника) (`FrameStore::FrameAt`); базовая частота — видео `source`,
  иначе самая медленная известная (исходник медленнее FG), иначе 1 fps (пассы без частоты). Окно префетча задаётся в
  секундах (`prefetch` кадров самого быстрого источника, у медленных — меньше кадров), `TexturesAt(t)` собирает кадр
  каждого источника на момент t, `ExpectedFrames` — сколько кадров нужно пассу до последнего кадра исходника в его
  частоте (FG на 48 fps: 479 из 479, а не «из 480»). Рендерер получает `FrameStates` базового слоя ячейки и рисует
  плашку вместо пустого фона: штриховка — «нет кадра», ровная — «загрузка…»; текст — в подписи ячейки
  (`source  #123 · 00:05.12 · нет кадра`). Таймлайн: таймкод `00:04.98 / 00:09.96`, кадр базового слоя, переключатель
  частоты при разных fps; шаг `,`/`.` — кадр базового слоя, воспроизведение — с максимальной частотой показанных
  источников; `dlssvid render --time SEC` (`--frame` — в базовой частоте). Корень «чёрных ячеек» аудита: скриншот снят
  на индексе 246 результата (48 fps), у 24-fps источников там не было кадра, а плашки не было; на временной оси кадр
  есть у каждого источника — регрессионный тест `[regression]` в `test_frame_store.cpp` (24 и 48 fps по всей шкале).
- **Сравнение одним действием (этап 9, MR C).** Панель над вьюпортом (`app/CompareBar`): режимы «До | После» (наложение
  двух полных слоёв + вертикальная шторка на 50 %, `ViewportState::SetCompare`), «Только после» (`SetAfterOnly`) и
  «Сетка 2×2»; режим выводится из состояния (`AppModel::compareView`: настроенная шторка → «До | После»), поэтому проект
  и CLI ничего нового не хранят. `W` включает «До | После», если шторка не настроена, иначе переключает её. Чипы слоёв —
  по одному на текущий источник с человеческим именем (`app/SourceNames`: Исходник, Глубина, Векторы, Апскейл,
  Улучшение, Генерация, Результат; технические имена — в подсказках) в порядке конвейера; клик по чипу меняет сторону
  «после» (или единственный вид), у чипа с историей — меню предыдущих версий. Предыдущие версии пасса (MR A) — источники
  вьюпорта с именем `<pass>@<id>` (`FrameStore::DiscoverPassVersions`, `Project::Sources`, `SplitSourceVersion`), их
  можно сравнивать шторкой и указывать в `render --layers`. Пресеты (меню «Вид»): Апскейл ↔ Улучшение, Исходник ↔
  Глубина; новый слой стека — 100 % и следующий неиспользованный источник (`NextUnusedSource`). «Инженерный режим»
  (кнопка панели, `Ctrl+E`, QSettings `view/engineerMode`) показывает стек слоёв и секции инспектора (слой, шторка,
  сетка); в обычном режиме в инспекторе остаётся только пробник.
- **Инцидент 2026-09-22 (остановка `FrameStore`).** Процесс юнит-теста в CI остался жить после пройденных проверок:
  главный поток в `~FrameStore` ждал loader, уснувший в `cv_.wait()` навсегда — `stop_` ставился без мьютекса, и loader
  между проверкой предиката и ожиданием пропускал `notify_all` (потерянное пробуждение, окно в несколько инструкций;
  гонка с этапа 4). Исправление: `stop_` под `mutex_`; регрессионный стресс-тест `[regression]` в `test_frame_store.cpp`
  (500 остановок с простаивающими и занятыми loader-потоками; до исправления зависал в 2 прогонах из 6).
- **Экран проекта (этап 9, MR D).** `core/stages/ParamSchema` — одно описание параметров каждой стадии (тип, диапазон,
  варианты с человеческими подписями, умолчание, `advanced`): `ValidateStageParams` проверяет параметры до запуска в
  `ProcessRunner::Prepare` (CLI `--param nr.intensity=9` → ошибка с текстом), `EffectiveStageParams` подставляет умолчания
  перед отпечатком — явное умолчание и отсутствующий ключ теперь один и тот же прогон (ограничение MR A снято), формы
  карточек строятся из схемы (`app/StageCard`: список бэкендов, переключатель масштаба/множителя/стабилизации, слайдер
  интенсивности; ключи `advanced` — только в JSON-редакторе инженерного режима). `ProcessOptions::forceStages` /
  `process --force nr,fg` пересчитывают выбранные стадии на месте (тот же отпечаток — без новой версии);
  `ProcessOptions::sourceHash` принимает готовый хэш — `Project::SourceHash()` кэширует sha256 исходника по размеру и
  mtime файла (`source_hash` в проекте), `process --project` и GUI не хэшируют неизменившийся файл заново. Проект хранит
  кодер результата (`encode.codec` / `encode.options`; `--codec` в CLI важнее). Раннер пишет `ms_per_frame` в манифест
  каждого пасса — `core/pipeline/Estimates` берёт его для оценки времени (иначе базовые мс/кадр реального прогона `face`
  из `docs/benchmarks.md`, масштабированные по мегапикселям) и считает `RunOutcome`: разрешение и частота результата,
  секунды того, что запустится, полный прогон, байты новых пассов. `AppModel::refreshPlan` (сразу после открытия и с
  задержкой 250 мс после правок) вызывает `PlanProcess` над проектом и раздаёт карточкам состояния
  (`AppModel::cardState`: «переиспользуется · 0 мин · посчитано вчера 14:02», «пересчёт: Интенсивность 1 → 1.4», «пересчёт:
  изменился вход · из-за «Улучшение»», «будет посчитано», «другой исходник», «досчитать…», «выключена»; тег версии vN;
  предупреждение «без глубины и векторов: качество ниже», когда у NR/FG/апскейла нет guides среди входов), панели
  «Что получится» (видео, время «к запуску k из m», совпавшие стадии, диск) и кнопке «Обработать · N стадий»
  (`Ctrl+Enter`, `dlssvid process --project … [--force …]`). Панель версий на диске — `ListPassVersions` с «Сравнить» /
  «Вернуть» / «Удалить» / «Очистить старые». Ошибка валидации показывается в плане вместо падения при запуске.
- **Старт, раскладка, сохранность (этап 9, MR E).** Центральная область — `QStackedWidget`: без проекта показан
  `app/StartPage` (что делает инструмент, зона перетаскивания видео, «Открыть видео…» / «Открыть проект…», строка
  готовности `AppModel::readiness` — адаптер, драйвер и какие DLL DLSS на месте, недавние файлы `AppModel::recents`
  из QSettings `recent/files` с состоянием проекта — «результат готов» / «N пассов» / «не обработан» / «файл не
  найден» и датой), доки при этом скрыты; с проектом — рабочая область, доки восстанавливаются из QSettings
  `window/state` (первый раз — все видны, док проекта 820 px). Геометрия окна — `window/geometry` (первый запуск —
  на весь экран), папки диалогов — `dialogs/<ключ>` (`lastDir` / `rememberDir`), панели возвращаются пунктами
  «Панель «…»» меню «Вид» (`toggleViewAction`), файлы открываются перетаскиванием на окно и через «Недавние».
  Несохранённые изменения: структурные правки состояния (слои, режимы, стадии, кодер) — `AppModel::markDirty`
  (`notifyStateChanged(false)` для зума, панорамы и шторки их не считает), «*» в заголовке, при закрытии
  `closeAction()`: `AutoSave` — проект сохраняется в свой файл (видео без проекта получает `<видео>.dlssvid.json`
  рядом), `Ask` — вопрос «Сохранить / Не сохранять / Отмена», когда файла нет.
- **Обработка и результат (этап 9, MR F).** «Обработать» ставит `dlssvid process --project` в `app/TaskQueue` вместе с
  планом стадий (`AppModel::outcome` → `TaskQueue::StagePlan`: имя, заголовок `AppModel::stageTitle`, оценка секунд,
  «переиспользуется») и открывает третью страницу центральной области — `app/ProcessingPanel`: «Прошло / Осталось»,
  общий прогресс по стадиям (переиспользованные засчитаны сразу), строка на стадию (готово с временем, бегущая с
  процентами, в очереди с оценкой; после сбоя «ошибка» только у упавшей стадии, остальные — «не запускалась»),
  «Отменить», «Свернуть в фон», хвост лога (8 строк без метки времени) и «Показать
  полный лог» (док «Лог»). Очередь разбирает строки CLI `stage: N/M frames` и сообщения раннера «process: <stage>
  done — … in S s» / «is complete, reused»: у бегущей стадии свой темп (после двух измеренных кадров; до того —
  оценка плана), у стадий в очереди — оценки, ETA = остаток бегущей + сумма очереди (`TaskQueue::Progress`).
  Отмена — `kill()` дочернего процесса (консольный CLI не реагирует на `terminate()`); пассы завершённых стадий уже
  имеют манифесты и переиспользуются следующим запуском, прерванная — `incomplete` в плане; задача в очереди
  снимается сразу. По завершении `MainWindow::onTaskFinished` перечитывает источники, включает «До | После»
  (`setCompareView`) и возвращает рабочую область; если окно свёрнуто или не активно — уведомление
  `QSystemTrayIcon::showMessage` (значок в трее живёт до возврата в окно), клик по нему разворачивает окно. Отмена
  и ошибка оставляют страницу обработки с объяснением и «К проекту»; закрытие окна во время обработки спрашивает
  и прерывает задачу. Процесс, который не запустился (`FailedToStart`), завершает задачу с текстом ошибки, а не
  висит в очереди. Открытие видео только ради `Info()` (`VideoDecoder::Probe()`: кэш кадров, план, `Prepare` раннера)
  логируется на уровне debug — в логе одна строка «open» на настоящее чтение. NVDEC внутри `process`: FFmpeg
  удерживает первичный CUDA-контекст и требует `CU_CTX_SCHED_BLOCKING_SYNC`, а стадия глубины (TensorRT, cudart)
  создаёт контекст раньше с планированием по умолчанию — `VideoDecoder` перед `av_hwdevice_ctx_create` выравнивает
  флаги `cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync)` (cudart применяет их и к активному контексту), иначе
  векторы декодировали бы программно и кормили OFA с CPU (регрессионный тест `[nvdec][regression]`).
- **Результат и его действия (TASK-0013).** Строка `app/ResultBar` под панелью сравнения появляется, когда у проекта
  есть видео результата (`AppModel::resultInfo` из источника «result» кэша кадров + проба звука при перечитывании,
  `resultSummary`: «3840×1600 · 48 fps · 479 кадров · со звуком · обновлён сегодня 19:23 · готов за 11:22» — время
  прогона этой сессии через `setLastRun`). Действия: «Сохранить как…» (`AppModel::exportResult` — копия файла,
  папка диалога `dialogs/result`), «Открыть папку» (`QDesktopServices`), «Экспорт пассов…» (по задаче `dlssvid export
  --from-dir <пасс> -o <папка>/<пасс> --format exr` на каждый текущий пасс с манифестом — `AppModel::exportablePasses`,
  очередь показывает прогресс), «Другое видео» = «Файл → Закрыть проект» (`Ctrl+W`): `MainWindow::settleUnsaved`
  (та же логика, что при закрытии окна) → `AppModel::closeProject` → стартовый экран; во время обработки закрыть
  проект нельзя. Клавиши `1…9` следуют порядку чипов панели сравнения (`Chip::hotkey`, подсказка чипа) и показывают
  источник как чип; «Что получится → Диск» дописывает свободное место на томе папки пассов (`AppModel::freeSpace`).
  Папка пассов меняется из блока «Источник» («Пассы · изменить…» → `AppModel::setPassesRoot`: проект помечается
  несохранённым, источники и план перечитываются из новой папки; не во время обработки); подписи ячеек несут размер
  источника («Исходник · 1920×800  #123 · 00:05.12»).
- **Битрейт кодирования (TASK-0016).** `core/io/EncodeDefaults`: без опции `b` FFmpeg-обёртка NVENC кодирует при своём
  низком умолчании (3840×1600 выходило ~1,4 Мбит/с), поэтому `EncodePassToVideo` и passthrough подставляют
  `RecommendedBitrateMbps(кодек, ширина, высота, fps)`: H.264 по точкам 720p30 ≈ 10, 1080p30 ≈ 16, 1440p30 ≈ 24,
  4K30 ≈ 45, 8K30 ≈ 160 Мбит/с (линейно между, +50 % от 30 к 60 fps, √ дальше), HEVC ≈ 0,65×, AV1 ≈ 0,55×, округление
  до «красивых» чисел; явное `b` (CLI `--codec-opt b=50M`, проект) остаётся как есть, FFV1 без битрейта. Карточка
  «Кодирование» в GUI: «Авто» (с показом значения), пресеты в Мбит/с, «Свой…», оценка размера файла из длительности
  источника (`ParseBitrateMbps` / `BitrateOption` — разбор и запись значения `b`).
- **Оформление по макетам (TASK-0017).** `app/Theme`: стиль Fusion с тёмной палитрой и один QSS по токенам макетов
  `Output/ux-mockups/*.dc.html` (фон и поля ввода `#17181b`, карточки `#1f2125`, рамки `#2c2f35` / `#3a3e46`, текст
  `#e6e7e9`, приглушённый `#a7abb3`, акцент `#4aa3df` с тёмным текстом `#0b1116`; шрифт Segoe UI 10 pt ≈ 13 px).
  Размеры из макетов: вторичные кнопки 36 px с отступом 14 px и радиусом 6, главное действие в акценте («Обработать»
  48 px / 16 px, старт 46 px / 15 px), табличные кнопки 24 px, сегменты 32 px в общей рамке, чипы 30 px с радиусом 14,
  списки и поля 30 px, карточки с полем 9×14 px, панели сравнения / результата 48 px с боковым отступом 16 px.
  Варианты задаются свойством `role` (`SetRole`): кнопки `primary`, `primary-big`, `start`, `start-primary`, `small`,
  `tall`, `quiet`; QToolButton `segment`, `chip`, `icon`; рамки `segments`, `card`, `card-dashed`, `dropzone`; подписи
  `muted`, `hint`, `title`, `h1`, `h2`, `lead`, `mono`, `ok`, `warn`, `danger`, `accent`, `version`, `cell-label`.
  Встроенных `setStyleSheet` с цветами в панелях больше нет — цвета только в теме.


### Апскейл (этап 5)

```
UpscaleStage ── YUV420P ─▶ Yuv420pToRgba16f ─▶ input RGBA16F ─┬─▶ [Resampler: сдвиг −j]  (только dlss)
                                                              ▼
   depth_dlss / mv_dlss (слот кэша или папки пассов) ─▶ IUpscaler::Evaluate(cmdlist) ─▶ output RGBA16F (UAV)
                                                              ▼
                                      readback ─▶ color_sr (EXR half / PNG16) + слот кэша + [--video через RgbToYuv420p]
IUpscaler: dlss (NGX, по умолчанию) | nis (NVScaler / NVSharpen, fallback) | bicubic (Catmull-Rom) | rtxvsr (опционально: RTX Video SDK, не основной с 2026-09-20)
```

- `ComputeKernel` — общий compute-помощник (root CBV + таблицы SRV/UAV + статические сэмплеры, кольца
  дескрипторов и констант); на нём `Resampler` (`Resample.hlsl`) и NIS (`Nis.hlsl` + `third_party/nis`).
- DLSS через NGX: `Init_with_ProjectID` (engine CUSTOM, DLL из `bin/nvidia/`, лог NGX → spdlog debug),
  проверка `SuperSampling.Available` с версией драйвера, `GET_OPTIMAL_SETTINGS` (режим по масштабу:
  ×1 DLAA, ≤1.55 Quality, ≤1.85 Balanced, ≤2.25 Performance, иначе Ultra Performance; вход должен попадать в
  диапазон render-размеров режима), флаги `DepthInverted` (reverse-Z `depth_dlss`) и `MVLowRes` (когда `mv_dlss`
  в разрешении входа), LDR-режим (вход display-referred 0..1), пресет через `DLSS.Hint.Render.Preset.*`,
  `InReset` на первом кадре. Без `depth_dlss`/`mv_dlss` — константная глубина и нулевые MV с предупреждением.
- Цель: `ResolveUpscaleTarget` — масштаб или явный размер, чётные стороны, пропорции, потолок 3840×2160 (v1).
- Стадия в пайплайне и в CLI один код (`RunUpscale`); в GUI стадия `upscale` запускается из панели проекта, `color_sr`
  появляется во вьюпорте как пасс (A/B в 2x2 и wipe, слои сэмплируются в своём разрешении).
- Метрики `passes/ImageMetrics` (PSNR Y/RGB, SSIM Y) и `dlssvid compare` — инструмент A/B для этапов 5–7.

### Neural Rendering (этап 6)

```
NrStage ── цвет: color_sr (папка / слот кэша) или декодированный кадр ─▶ RGBA16F ─▶ Tonemapper (passthrough | aces | reinhard)
   ▼                                                                                   │
 proxy (полное разрешение) ──[model-scale ≠ 1: Resampler → work]──▶ INrBackend × passes ─▶ model out (work)
   ▼                                                                                   │
 NrCompose::Resolve(original = proxy, model, proxy-work, prev, mv_dlss, mask protect / skin) ─▶ output RGBA16F
   ▼
 readback ─▶ color_nr (EXR half / PNG16) + слот кэша + [--video]
INrBackend: ngx (nvngx_dlssnr.dll, NGX Feature 18) | stub (детерминированная правка для тестов, только явно)
```

- `Tonemapper` (`Tonemap.hlsl`): вход `srgb | linear | pq | hlg`, экспозиция, кривая `passthrough | aces | reinhard`, выход
  display-referred sRGB [0, 1] — то, что принимает модель (ComfyUI подаёт clamp [0,1]). Для SDR-источников по умолчанию
  passthrough (точная копия); PQ/HLG ждут 10-битного декодера. CPU-эталон `TonemapReference` — для тестов.
- `NgxNrBackend` — Feature 18 по образцу ComfyUI-DLSS5-NR (последовательность вызовов) и OptiScaler_DLSSNR (имена параметров):
  1. диагностика **до** любого вызова: GPU, архитектура (CUDA compute capability, иначе по имени), версия драйвера из UMD-версии
     DXGI (`32.0.15.9186` → 591.86; порог **616.56**, `--skip-driver-check`), путь + размер + **SHA-256** `nvngx_dlssnr.dll`,
     путь форвардера — всё в лог и в `NrDiagnostics` (`dlssvid nr --check`, `--json`);
  2. ядро NGX через общий `gpu/Ngx` (`Init_with_ProjectID`, рефкаунт на процесс; тот же, что у DLSS SR);
  3. `LoadLibraryEx(nvngx_dlssnr.dll)` + форвардер `nvngx.dll_dlssvid.dll` (см. хаки): `Init_Ext` сниппета с перебором ABI
     (`info, version` ComfyUI → `version, info` публичный → `version, params` OptiScaler), блок параметров `capability`
     (OptiScaler) или `alloc` (ComfyUI) с автоповтором;
  4. все `DLSSNR.*` пишутся перед `CreateFeature(18)` (тюнинг читается при создании) и заново перед каждым `EvaluateFeature`
     (блок общий): `Enabled`, `Width/Height`, `Hint.Render.Preset`, `Intensity`, `Style`, `Local{Tone,Structure}Strength`,
     `SkinStructureStrength`, `UseAutoMask`, `UICorrection`, `DepthInverted`, `Reset`, `ScalingRatio`, ресурсы
     `Color/Output/Backbuffer/Depth/MVec` с субректами, `MVecScaleX/Y`;
  5. `CreateFeature(18)` в `Init` со scratch-текстурами — результат в лог; `FAIL_FeatureNotSupported` на не-Blackwell →
     подсказка `dlssvid nr-patch`; `FAIL_PlatformError` → диагностика форвардера.
  Guides: `depth_dlss` (R32F, reverse-Z → `DepthInverted=1`) и `mv_dlss` (RG32F, пиксели, backward → `MVecScale=1`) субректами в
  своём разрешении; без MV — «still»-режим (`Reset=1` на каждом кадре, как ComfyUI), с MV — temporal (`Reset` на первом кадре).
- `NrStage`: цвет из `--color-dir` (по умолчанию `<passes>/color_sr`, если есть), слота кэша (`color_sr`) или декодированного
  кадра; guides из папок → слот кэша → текстуры; маски `mask_{ui,ignore,face,skin}` (PNG 8-бит, ТЗ §5) → `protect = max(ui,
  ignore)`, `skin = max(face, skin)`; `--passes 2` — вторая копия бэкенда со своей историей поверх результата первой;
  `--model-scale` — модель на уменьшенном/увеличенном прокси, `NrCompose` переносит правку на полный кадр. Недоступный
  бэкенд: CLI — ошибка с инструкцией (код 1); пайплайн (`disableWhenUnavailable`) — стадия отключается с предупреждением,
  кадры идут дальше без `color_nr` (ТЗ §4).
- `NrPatch` / `dlssvid nr-patch`: обёртка внешнего dlssnr-patcher (GPLv2 — не вендорится): python, скрипт
  (`--patcher`, `DLSSNR_PATCHER_ROOT`, `tools/dlssnr-patcher`), CUDA 13.3 (`--cuda-bin`, `CUDA_PATH_V13_3`), SHA-256 входа и
  результата, `bin/nvidia/nvngx_dlssnr.dll` + сайдкар `*.patch.json` (что и чем пропатчено). GUI: кнопка «Пропатчить DLL…».
- Стадия `nr` в проекте по умолчанию (после `upscale`); `color_nr` во вьюпорте — A/B в 2x2 и wipe.

### Frame Generation (этап 7)

```
FgStage ── цвет: color_nr / color_sr (папка или слот) или декодированный кадр ─▶ RGB F16 (CPU) + RGBA16F (GPU, ping-pong prev/cur)
   ▼
 IFrameGenerator::Generate(prev, cur, depth_dlss, mv_dlss, reset) ─▶ multiplier − 1 кадров (RGB F16, CPU)
   ▼
 color_fg: real i → i·mult, generated → i·mult + k; манифест fps × mult; [--video с fps × mult]
IFrameGenerator: dlssg (NGX Feature 11, DLSS SDK) | rife (RIFE 4.x, TensorRT) | blend (lerp, CPU)
```

- **Решение: DLSS-G через NGX, а не Streamline.** ТЗ §4 планировало Streamline со скрытым swapchain и отдельный
  процесс `fg_worker`, потому что Streamline генерирует кадры только в перехваченном `Present`. DLSS SDK 310.9
  документирует прямой путь (`nvsdk_ngx_helpers_dlssg_d3d.h`, «DLSS-FG Programming Guide»): `NGX_D3D12_CREATE_DLSSG`
  (размер, формат backbuffer, `RenderWidth/Height` = размер guides) и `NGX_D3D12_EVALUATE_DLSSG` с `Backbuffer`
  (текущий кадр), `Depth`, `MVecs` (пиксели в разрешении MV, «от текущего к предыдущему» — наш `mv_dlss` со
  `mvecScale = 1`), выход `OutputInterpolated` — кадр между **предыдущим** backbuffer (его хранит рантайм) и текущим;
  `multiFrameCount/Index` — evaluate вызывается `mult − 1` раз на пару. Ни swapchain, ни хуков, ни бинарников Streamline
  (в клоне с GitHub их и нет). Изоляция для GUI осталась процессной: панель проекта запускает `dlssvid fg` через
  `TaskQueue`, падение процесса — `taskFinished(false)` (тест с `--crash-after`).
- `DlssgFrameGenerator`: capability check (`FrameGeneration.Available`, `MultiFrameCountMax`, версия драйвера из блока),
  создание с `rgba16f`, при отказе — `rgba8` через compute-конверсию; guides через субректы в своём разрешении, без них —
  константная глубина / нулевые MV с предупреждением; первый кадр — `reset`, его выход отбрасывается (рантайм
  «запоминает» кадр). Лог как у SR/NR (GPU, DLL + SHA-256, результат создания).
- `RifeFrameGenerator`: реестр моделей → ONNX (`models/cache/`) → `TrtEngine` с профилем фиксированной формы
  (`Options::shapes`, min = opt = max, форма в ключе кэша) под размер, паддингованный до кратного 32; NCHW RGB [0,1],
  `timestep = k/mult` → ×2/×3/×4 одной моделью; выход обрезается.
- `BlendFrameGenerator`: `lerp(prev, cur, k/mult)` — наивный baseline и детерминированный бэкенд для тестов на WARP.
- `dlssvid compare --start 1 --step 2` — только сгенерированные кадры против выброшенных оригиналов (ground truth,
  `docs/benchmarks.md`).

### Полный пайплайн, пакет и golden-тесты (этап 8)

```
dlssvid process ── ProcessRunner::RunProcess ── depth ─▶ flow ─▶ upscale ─▶ nr ─▶ fg  (RunDepth/RunFlow/RunUpscale/RunNr/RunFg над <passes>/)
                                             │   пасс полный (manifest.frameCount ≥ нужного) ─▶ reused
                                             ▼
                          EncodePassToVideo(color_fg › color_nr › color_sr) + аудио источника ─▶ result
```

- Стадии запускаются по очереди над дисковым кэшем пассов (ТЗ §4): каждая читает пассы предыдущих (`depth_dlss`,
  `mv_dlss`, `color_sr`, `color_nr`, `mask_*`) из корня, пишет свой; полный пасс переиспользуется (`--no-skip-existing`
  пересчитывает), недоступные `nr`/`fg` — ошибка или `--disable-unavailable` (стадия пропускается с причиной, отчёт).
  Параметры стадий — те же JSON-ключи, что в файле проекта и в GUI: `RunUpscale/RunNr/RunFg` получают их через
  `StageConfig` (`ApplyParams`), depth/flow — маппинг в раннере (`depth.model`, `flow.perf`, …); `--param stage.key=value`
  и алиасы `--scale`, `--multiplier`, `--<stage>-backend`.
- `EncodePassToVideo`: кадры пасса с его fps; источник декодируется параллельно только ради аудио-пакетов
  (`SetAudioPacketSink` → `WriteAudioPacket`), `mult = round(passFps / sourceFps)` кадров пасса на кадр источника —
  видео и аудио пишутся вперемежку. Без цветовых стадий — passthrough (этап 0).
- `dlssvid bench` — тот же `RunProcess` во временный корень без переиспользования, отчёт мс/кадр по стадиям + GPU /
  архитектура / драйвер (JSON) — `docs/benchmarks.md`.
- GUI: кнопка «Обработать → result» (`ProjectPanel::processAll`) сохраняет проект и ставит `dlssvid process --project`
  в `TaskQueue`; `result` затем появляется как источник вьюпорта.
- Golden-тесты (`tests/golden/`): детерминированный прогон на WARP (стабы + NIS + blend) сверяется с эталонными кадрами
  и ожиданиями (`expected.json`, `ref/`), плюс `[gpu]` прогон реальных бэкендов; `DLSSVID_CLI`/`DLSSVID_GOLDEN_DIR`
  направляют тест на установленную копию (`docs/release.md`).
- Пакет: `install()` копирует `build/<preset>/bin` (без `*.pdb`, тестов и **без `bin/nvidia/*.dll`**), реестр и скрипты
  моделей, `depth_worker`, документацию и golden-данные; CPack ZIP всегда, NSIS при наличии `makensis`; CI — `.gitlab-ci.yml`
  + `scripts/ci-build.cmd` (`docs/ci.md`).
- **Гигиена сборки (инцидент этапов 4 и 8):** ninja берёт зависимости от заголовков из строк `/showIncludes`, сверяя их с
  `msvc_deps_prefix`, который CMake записал при configure. У локализованного MSVC (русская VS без английского языкового
  пакета) байты префикса и вывода компилятора совпадают не во всех окружениях (кодовые страницы консоли), и объект
  получает «#deps 0» — он молча переживает правку заголовка. Ctest `build.ninja_header_deps`
  (`tests/check_ninja_deps.cmake`) проверяет после сборки, что у каждого объекта с проектными включениями есть
  зависимости; пресеты задают `VSLANG=1033` (действует после установки английского пакета); CI собирает с нуля.

### Отпечатки и версии пассов (этап 9, MR A)

```
PlanProcess / RunProcess ── для каждой включённой стадии ── DecideStage(что на диске, что решено выше)
   отпечаток = sha256(канонический JSON {source, stage, params, inputs: {пасс → отпечаток}, tool: {app, backend, model, model_hash, dll_file, dll}});
   `tool.app` — версия алгоритма пассов `kPassToolVersion` (с 0.2.0 не равна версии релиза: релиз не обесценивает пассы)
   манифест <root>/<pass>/:  совпал и полный ─▶ reuse (exact)          │ совпал, кадров мало ─▶ run (incomplete, на месте)
                             без отпечатка   ─▶ reuse (legacy, штамп)  │ другой ─▶ такая версия в истории ─▶ restore
                                                                       │         иначе run (source_changed | params_changed | input_changed | tool_changed) + retire
   run: <root>/<pass>/ ─▶ <root>/<pass>.v/<YYYYMMDD-HHMMSS_<fp8>>/, стадия пишет на место, манифест получает
        fingerprint / inputs / tool / params_canonical / created; после прогона gc (keepVersions, по умолчанию 2)
```

- `core/pipeline/PassFingerprint`: `CanonicalizeJson` (ключи отсортированы, `2.0 == 2`, строка ≠ число),
  `ToolForStage` — идентичность инструмента до запуска: бэкенд после умолчаний стадии, id модели
  (`TrtDepthEstimator::DefaultModel`, `flow.model`, `fg.model`) и его sha256 из `models/registry.json`, NVIDIA DLL,
  которую бэкенд загрузит (`NvidiaDllSearchPaths` / `FindNrDll` / `FindDlssgDll`), её sha256 кэшируется по
  (путь, размер, mtime). Таблицы стадий (`StageOutputPasses`, `StageFamilyPasses`, `StageInputCandidates`) повторяют
  `ExistingPass()` раннера: flow читает `depth_raw`, upscale — `depth_dlss`/`mv_dlss`, nr — `color_sr` + guides +
  `mask_*`, fg — `color_nr`, иначе `color_sr`.
- `core/pipeline/PassVersions`: текущая версия остаётся в `<root>/<pass>/` (ни один читатель не меняется), история —
  `<root>/<pass>.v/<id>/`, id = `created` манифеста (иначе mtime) + 8 символов отпечатка (`legacy` без него,
  `partial` для папки без манифеста). `RetirePassVersion` / `UsePassVersion` (обмен папок, id или уникальный префикс) /
  `ListPassVersions` / `GcPassVersions`: держит `keep` новейших на пасс (текущая считается), не трогает версии, на
  которые ссылаются `inputs` любой версии любого пасса; обходит пассы от конца конвейера (снятая версия `color_fg`
  освобождает версию `color_nr` в том же вызове), dry-run имитирует удаления.
- `ProcessRunner`: одно решение `DecideStage` для плана и прогона. План идёт по проекции (отпечатки, которые стадии
  выше произведут), прогон перечитывает диск перед каждой стадией — отключённый `nr` (`--disable-unavailable`)
  оставляет `fg` читать `color_sr`, и это попадает в отпечаток `color_fg`. Штамп ставится после успешного завершения
  стадии на все папки её семейства (`depth_raw` + `depth_dlss`); упавшая или отключённая стадия штамп не получает,
  снятая версия остаётся в истории. Число кадров в отпечаток не входит — это проверка полноты (`incomplete` →
  пересчёт на месте, версия не создаётся); `--no-skip-existing` при равном отпечатке тоже пишет на место.
  Пассы без отпечатка (этап 8, `dlssvid <stage>` вручную) усыновляются при первом прогоне — только если совпадает хэш
  исходника и ни один их вход в этом плане не пересчитывается; иначе пересчёт.
- Манифест: блок `fingerprint`, `inputs`, `tool`, `params_canonical`, `created` (`docs/conventions.md` §6),
  schema_version прежняя; блок пишет только `dlssvid process`. Отчёт `process --json`: `fingerprint`, `retired`,
  `restored`, `decision` у каждой стадии; `process --plan --json` — план целиком.
- CLI: `process --plan`, `--keep-versions N` (проект: `pass_versions_keep`, по умолчанию 2, 0 = хранить всё);
  `passes list|use|gc` (`cli/PassesCommands`). GUI подключается в MR D (карточки стадий показывают решения плана).
- Границы: версии итогового видео не ведутся; хэш DLL — той, что найдена в момент плана. (С MR D параметры хэшируются с
  умолчаниями схемы: явное умолчание и отсутствующий ключ — один прогон.)

## Хаки и временные решения

| Где | Что | Почему | Когда убираем |
|---|---|---|---|
| `VideoDecoder::ReceiveFrame` | NVDEC-кадры скачиваются на CPU (`av_hwframe_transfer_data`) | этап 0 проверяет bit-exact путь, а не производительность | этап 2: NVDEC → CUDA → D3D12 без CPU |
| `VideoDecoder::ConvertFrame` | не-8-бит-4:2:0 источники → swscale → yuv420p (lossy) | стадии работают в 8-бит 4:2:0 до появления GPU-конверсии | этап 2: RGBA16F на GPU |
| `PassthroughStage` | round-trip upload → readback каждого кадра | доказательство корректности пути GPU | остаётся как диагностический режим |
| `MvConvert::InvertFlow` | инверсия flow сплэттингом с округлением до пикселя, дыры — BFS-заполнением | простая детерминированная инверсия без субпиксельного ресемплинга | этап 3: сравнить с backward-warp SEA-RAFT/OFA, при необходимости заменить |
| `ColorConvert` | хрома 4:2:0 реплицируется (nearest), без интерполяции | детерминизм и обратимость поблочно | этап 3+: конверсия YUV → RGB на GPU с настраиваемым фильтром |
| `DepthStage`, `TrtFlowEstimator` | препроцессинг/апсемпл/guided filter на CPU; входы TensorRT копируются host→device | модели и контракты проверяются на CPU-пути; шейдерной инфраструктуры ещё нет | этап 4–5: препроцессинг compute-шейдером на общем D3D12-девайсе, вход TensorRT из CUDA-буфера |
| `FlowStage::Process` | NV12-кадр копируется в собственный device-буфер стадии | декодер отдаёт один и тот же буфер на следующем кадре | этап 4+: кольцо NVDEC-кадров в `GpuFrameCache` |
| `TemporalAlignmentError` | статичный прокси, если `--mv-dir` не задан | обратная совместимость | — (warp-версия есть) |
| `FrameStore::LoadVideo` | видео для вьюпорта декодируется на CPU и конвертируется в RGBA16F (`Yuv420pToRgba16f`, F16C) перед upload | NVDEC-кольцо и YUV→RGB на GPU появятся вместе с GPU-препроцессингом этапа 5; CPU-путь укладывается в ≥ 30 fps на 1080p | этап 5: NVDEC → CUDA → D3D12-текстура, конверсия compute-шейдером |
| `ViewportRenderer::RenderInto` | каждый кадр — `ExecuteAndWait` (синхронный submit) | простота; composite 0.2–2.5 мс, узкое место — загрузка кадров | этап 5+: fence-ринг и несколько кадров в полёте |
| `ViewportWindow` | подписи ячеек и пробник — виджеты Qt рядом с вьюпортом, в PNG-скриншот их вписывает `QPainter` | Qt не рисует поверх дочернего нативного окна со swapchain | — (по плану этапа 4) |
| `UpscaleStage` + `DlssUpscaler` | **эмуляция джиттера** (ТЗ §3): кадр видео пересемплируется Catmull-Rom со сдвигом −j (Halton(2,3), фаз = 8·(target/render)²), содержимое смещается на +j, DLSS получает `InJitterOffset = +j`; MV не джиттерятся (`MVJittered = 0`) | DLSS SR рассчитан на джиттерный рендер, у видео джиттера нет; знак выбран по PSNR на синтетическом эталоне (+1.7 дБ к прогону без джиттера, `--jitter-sign`) | остаётся как режим сравнения; RTX VSR — основной путь |
| `RtxVsrUpscaler` | заглушка: `Available()` объясняет, что нужен RTX Video SDK 1.1, стадия деградирует в `nis` | SDK только под аккаунтом NVIDIA Developer, которого нет — RTX VSR снят как основной бэкенд (DECISION 2026-09-20), по умолчанию `dlss` | интеграция за `RTX_VIDEO_SDK_ROOT`, если SDK появится |
| `UpscaleStage::Process` | результат читается на CPU (readback) для записи пасса и заново грузится в слот кэша | простота; следующие стадии пока читают пассы с диска | этап 6–7: копия текстуры в слот без CPU, запись EXR в фоне |
| `NisUpscaler` | масштаб > 2 — два прохода NVScaler (×2, затем остаток) | NIS принимает соотношение 1..2 за проход | — |
| `VideoEncoder` (превью `--video`) | цветовые теги (matrix/range) не записываются в поток | превью без аудио, для просмотра; финальный мукс — этап 8 `process` | этап 8: теги и аудио |
| `core/stages/nr/forwarder/` (`nvngx.dll_dlssvid.dll`) | **форвардер**: все вызовы в `nvngx_dlssnr.dll` (`Init_Ext`, `CreateFeature`, `EvaluateFeature`, `ReleaseFeature`) идут из отдельной DLL, в имени которой есть `nvngx.dll`, без tail-call (`volatile`, `noinline`) | модель проверяет модуль по адресу возврата и отвечает `FAIL_PlatformError` любому другому (OptiScaler_DLSSNR, ComfyUI caller shim); proxy-путь через ядро даёт `0xBAD0000B` | когда NVIDIA откроет Feature 18 публичным SDK / Streamline |
| `NgxNrBackend::Init` | перебор ABI `Init_Ext` сниппета (три порядка аргументов) и двух блоков параметров с автоповтором `CreateFeature(18)` | эталоны расходятся; на RTX 4070 Ti SUPER с DLL 310.8 (Ada-патч) сработали первый вариант (порядок ComfyUI) и capability-блок — перебор оставлен как страховка для других сборок DLL | когда появится публичный SDK Feature 18 |
| `NrCompose` (`model-scale ≠ 1`) | **перенос отношением**: `edited = original · clamp(model↑ / proxy↑, 1/maxRatio, maxRatio)` с силой `transfer` | упрощённый resolve OptiScaler (без OkLab и разделения яркость/цвет): модель на уменьшенном прокси дешевле, детали полного разрешения остаются | если модель примет `ScalingRatio` напрямую |
| `NrCompose` (`--temporal`) | **temporal-фильтр**: смешивание с предыдущим выходом, смещённым по `mv_dlss`, с порогом по разности цвета и весом правки | мерцание пропатченных DLL (ТЗ §4 «известная проблема»); опция, по умолчанию выключена | когда guides уберут мерцание |
| `StubNrBackend` | детерминированная «правка» (локальный контраст + тон) вместо модели | стадия, resolve, маски, temporal и pass count тестируются на WARP без DLL; в продукте только явно `--backend stub`, без fallback | — (остаётся тестовым бэкендом) |
| `NrStage::Process` | результат читается на CPU (readback) для записи пасса и заново грузится в слот; guides и маски грузятся с диска каждый кадр | простота; как в `UpscaleStage` | этап 8: слот → слот без CPU |
| `INrBackend` (драйвер) | версия драйвера NVIDIA из UMD-версии DXGI (`CheckInterfaceSupport`), а не из NVAPI | NVAPI не подключён; формула `(subversion mod 10)·100 + build/100` проверена на 591.86 | при подключении NVAPI |
| `DlssgFrameGenerator` (камера) | **синтетические константы камеры** для DLSS-G: перспектива FOV 60°, near 0.1 / far 1000 (`BuildFgCamera`), `clipToPrevClip = prevClipToClip = I`, `cameraMotionIncluded = 1`, `menuDetectionEnabled = 0`, `motionVectorsInvalidValue = FLT_MAX` | у видео нет камеры; всё движение — в `mv_dlss`, как у DLSS5-Feeder для игр без DLSS | если модель заметно выиграет от оценки движения камеры (этап 8: оценка из MV) |
| `DlssgFrameGenerator` (backbuffer) | цвет подаётся как `R16G16B16A16_FLOAT` с `ColorBuffersHDR = 0`; при `FAIL_UnsupportedFormat` — копия в `R8G8B8A8_UNORM` через `Resampler` | руководство описывает display-ready SDR/HDR10; RGBA16F [0,1] приняты на Ada | — |
| `FgStage` / `IFrameGenerator::Generate` | каждый сгенерированный кадр читается на CPU (readback / TensorRT host) и пишется в пасс; `color_fg` не кладётся в слот кэша | у `color_fg` больше кадров, чем у источника (нет соответствия «один слот — один индекс») | этап 8: `process` с потоковой записью |
| `RifeFrameGenerator` | вход паддится до кратного 32 (replicate), движок под паддингованный размер | RIFE 4.x работает на кратных 32 | — |
| `FgStage --crash-after` | скрытый флаг: `TerminateProcess(0xC0000005)` после N кадров | воспроизводимое «падение воркера» для теста изоляции (ТЗ §9) | остаётся тестовым |
| `EncodePassToVideo` | источник декодируется целиком ради аудио-пакетов | у `VideoDecoder` нет режима «только аудио»; декодирование дёшево по сравнению со стадиями | этап 8b: демукс аудио без декодирования видео |
| `ProcessRunner` | стадии выполняются последовательно через дисковый кэш пассов, а не одним потоковым пайплайном | дисковый кэш нужен по ТЗ §4, и каждая стадия уже умеет читать пассы; потоковый режим потребовал бы убрать CPU-readback | — |

Хаки в коде помечаются `// HACK:` (ТЗ §8). Скрытый swapchain для FG не понадобился (прямой NGX-API, см. этап 7).

## Ошибки и деградация
- Отсутствие NVIDIA GPU: D3D12 падает на WARP, CUDA interop недоступен, NVENC-кодеки не найдены —
  всё сообщается в логе и `dlssvid info`; `--codec ffv1` и `--warp` работают везде.
- Ошибки FFmpeg/D3D12/CUDA — исключение `dlssvid::Error` с кодом и текстом; CLI печатает и возвращает 1.
