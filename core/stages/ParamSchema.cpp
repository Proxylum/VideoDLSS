#include "stages/ParamSchema.h"

#include "passes/ControlMaskPass.h"

#include "util/Error.h"

#include <algorithm>

#include <cmath>
#include <cstdio>

namespace dlssvid {

namespace {

using J = nlohmann::json;

ParamSpec Enum(const char* key, const char* label, const char* def, std::vector<ParamChoice> choices, ParamWidget widget = ParamWidget::Select,
               const char* hint = "", bool advanced = false) {
    ParamSpec s;
    s.key = key;
    s.type = ParamType::Enum;
    s.widget = widget;
    s.label = label;
    s.hint = hint;
    s.def = def;
    s.choices = std::move(choices);
    s.advanced = advanced;
    return s;
}

ParamSpec Number(const char* key, const char* label, ParamType type, J def, double min, double max, double step, ParamWidget widget, const char* hint = "",
                 bool advanced = false, std::vector<ParamChoice> choices = {}) {
    ParamSpec s;
    s.key = key;
    s.type = type;
    s.widget = widget;
    s.label = label;
    s.hint = hint;
    s.def = std::move(def);
    s.min = min;
    s.max = max;
    s.step = step;
    s.advanced = advanced;
    s.choices = std::move(choices);
    return s;
}

ParamSpec Flag(const char* key, const char* label, bool def, const char* hint = "", bool advanced = true) {
    ParamSpec s;
    s.key = key;
    s.type = ParamType::Bool;
    s.widget = ParamWidget::Check;
    s.label = label;
    s.hint = hint;
    s.def = def;
    s.advanced = advanced;
    return s;
}

ParamSpec OnlyFor(ParamSpec s, std::vector<std::string> backends) {
    s.backends = std::move(backends);
    return s;
}

ParamSpec WithFormChoices(ParamSpec s, std::map<std::string, std::vector<ParamChoice>> lists) {
    s.formChoices = std::move(lists);
    s.widget = ParamWidget::Select;
    return s;
}

ParamSpec OmitDefault(ParamSpec s) {
    s.omitDefault = true;
    return s;
}

// an engineer key a pipeline preset holds (PR-9): of the algorithm, not of the machine
ParamSpec Tuning(ParamSpec s) {
    s.tuning = true;
    return s;
}

// form placement (TASK-0046)
ParamSpec In(ParamSpec s, const char* group, bool more = false) {
    s.group = group;
    s.more = more;
    return s;
}

ParamSpec ShowIf(ParamSpec s, const char* key, ParamCondition::Op op, J value) {
    s.showIf.push_back({key, op, std::move(value)});
    return s;
}

ParamSpec EnableIf(ParamSpec s, const char* key, ParamCondition::Op op, J value, const char* note) {
    s.enableIf.push_back({key, op, std::move(value)});
    s.disabledNote = note;
    return s;
}

ParamSpec NeedsInputs(ParamSpec s, std::vector<std::string> inputs) {
    s.needsInputs = std::move(inputs);
    return s;
}

// Disabled with `note` while one of these passes is an input of the plan (PR-8: the control mask replaces the automatic one); RowState
// checks it before enableIf, so this note wins over the row's own condition.
ParamSpec DisabledByInputs(ParamSpec s, std::vector<std::string> inputs, const char* note) {
    s.disabledByInputs = std::move(inputs);
    s.disabledByInputsNote = note;
    return s;
}

ParamSpec Display(ParamSpec s, const char* zeroLabel, const char* suffix = "") {
    s.zeroLabel = zeroLabel;
    s.suffix = suffix;
    return s;
}

ParamSpec Decimals(ParamSpec s, int decimals) {
    s.decimals = decimals;
    return s;
}

// The VDA models of models/registry.json (family vda, metric): "" is the backend's default, metric-vda-small.
std::vector<ParamChoice> VdaModels() {
    return {{"", "Small — Apache-2.0", "Metric Video Depth Anything Small: 28M параметров, окно 32 кадра; по умолчанию"},
            {"metric-vda-base", "Base — CC-BY-NC-4.0 (некоммерческая)", "113M параметров; ≈ 5 ГБ видеопамяти на 1080p, +33–42 % времени к Small; только некоммерческое использование"},
            {"metric-vda-large", "Large — CC-BY-NC-4.0 (некоммерческая)",
             "382M параметров; ≈ 10 ГБ видеопамяти на 1080p при окне 32, +45–65 % времени; только некоммерческое использование; первый запуск — экспорт и сборка TensorRT ≈ 11–14 мин"}};
}

ParamSpec Text(const char* key, const char* label, const char* def, const char* hint = "", bool advanced = true) {
    ParamSpec s;
    s.key = key;
    s.type = ParamType::String;
    s.widget = ParamWidget::Text;
    s.label = label;
    s.hint = hint;
    s.def = def;
    s.advanced = advanced;
    return s;
}

std::vector<ParamChoice> Formats(bool png) {
    std::vector<ParamChoice> c{{"exr", "EXR (half)", ""}, {"tiff", "TIFF", ""}, {"npz", "NPZ", ""}, {"raw", "raw", ""}};
    if (png) c.insert(c.begin() + 1, ParamChoice{"png", "PNG 16 бит", ""});
    return c;
}

std::vector<StageSchema> Build() {
    std::vector<StageSchema> out;
    {  // PR-13 (spec §2d): the clip's scene-cut list, read by the temporal stages
        StageSchema s;
        s.stage = "scenes";
        s.title = "Склейки";
        s.subtitle = "где стадии начинают историю заново · проход по яркости исходника";
        s.params = {
            In(Display(Number("threshold", "Порог склейки", ParamType::Float, 0.24, 0.0, 1.0, 0.01, ParamWidget::Spin,
                              "Средняя разница яркости соседних кадров (0…1), выше которой начинается новый план. На склейке векторы движения "
                              "нулевые, DLSS SR и улучшение начинают историю заново, генерация кадров повторяет последний кадр плана вместо "
                              "смешивания двух сцен. 0.24 ловит резкие склейки; яркая вспышка может дать лишний сброс. Смена порога пересчитывает "
                              "стадии ниже, только если меняется список. 0 — выкл: стадии работают без сбросов."),
                       "выкл"),
               "Обнаружение"),
            Enum("backend", "Метод", "luma", {{"luma", "яркость исходника", "средняя разница яркости соседних кадров"}}, ParamWidget::Select, "", true),
        };
        out.push_back(std::move(s));
    }
    {
        StageSchema s;
        s.stage = "depth";
        s.title = "Глубина";
        s.subtitle = "guide для DLSS, NR и FG";
        s.params = {
            Enum("backend", "Метод", "da3",
                 {{"da3", "DA3 metric (точнее)", "Depth Anything 3, метрическая глубина"},
                  {"vda", "VDA (стабильнее)", "Video Depth Anything: меньше мерцания; модель — ниже"},
                  {"worker", "DA3 через PyTorch", "эталонный путь depth_worker (models/export/.venv)"},
                  {"worker:vda", "VDA через PyTorch", "эталонный путь depth_worker, модели те же"},
                  {"stub", "stub (тест)", "синтетическая глубина без модели"}}),
            Enum("stabilize", "Стабилизация", "auto",
                 {{"auto", "авто", "metric-модели (DA3, VDA metric) — масштаб, относительные — масштаб и сдвиг"},
                  {"none", "выкл", "глубина кадр за кадром, как её дала модель"},
                  {"scale", "масштаб", "только масштаб: метры сохраняют отношения"},
                  {"scale_shift", "масштаб + сдвиг", "для относительной глубины; на метрической сдвиг искажает метры"}},
                 ParamWidget::Toggle, "временное выравнивание глубины по окну ±N кадров вокруг кадра (N — «Окно стабилизации»)"),
            WithFormChoices(Text("model", "Модель", "", "пусто = модель бэкенда по умолчанию", false), {{"vda", VdaModels()}, {"worker:vda", VdaModels()}}),
            OmitDefault(OnlyFor(Number("window", "Окно кадров VDA", ParamType::Int, 0, 0, 32, 4, ParamWidget::Spin,
                                       "кадров за проход модели, кратно 4; 0 = как в реестре (32); меньше окно — меньше видеопамяти", false),
                                {"vda", "worker:vda"})),
            Number("input_size", "Размер входа модели", ParamType::Int, 518, 64, 4096, 14, ParamWidget::Spin, "", true),
            Number("max_res", "Максимальная сторона", ParamType::Int, 1080, 256, 8192, 1, ParamWidget::Spin, "", true),
            Tuning(Number("stabilize_window", "Окно стабилизации, ± кадров", ParamType::Int, 8, 1, 64, 1, ParamWidget::Spin,
                          "кадр выравнивается по кадрам t−N…t+N; глубина выходит с задержкой N кадров", true)),
            Tuning(Number("z_near", "Ближняя плоскость, м", ParamType::Float, 0.1, 0.001, 100.0, 0.01, ParamWidget::Spin, "", true)),
            Tuning(Number("z_far", "Дальняя плоскость, м", ParamType::Float, 1000.0, 1.0, 100000.0, 1.0, ParamWidget::Spin, "", true)),
            Flag("fp32", "FP32", false, "модель в fp32 вместо fp16"),
            Tuning(Flag("no_fill", "Не заполнять дыры", false)),
            Tuning(Flag("no_dlss", "Только depth_raw", false, "не писать depth_dlss")),
            Tuning(Enum("format", "Формат пасса", "exr", Formats(false), ParamWidget::Select, "", true)),
            Text("models_dir", "Папка моделей", ""),
            Text("python", "Python воркера", ""),
        };
        out.push_back(std::move(s));
    }
    {
        StageSchema s;
        s.stage = "flow";
        s.title = "Векторы движения";
        s.subtitle = "guide для DLSS и FG";
        s.guides = {"depth_raw"};
        s.params = {
            Enum("backend", "Метод", "ofa",
                 {{"ofa", "Optical Flow (NVIDIA, точнее)", "NVIDIA Optical Flow SDK"}, {"searaft", "SEA-RAFT", "TensorRT"}, {"stub", "stub (тест)", ""}}),
            Enum("perf", "Качество", "slow", {{"slow", "высокое", ""}, {"medium", "среднее", ""}, {"fast", "быстрое", ""}}, ParamWidget::Toggle,
                 "уровень производительности Optical Flow"),
            Tuning(Text("model", "Модель (id реестра)", "", "пусто = модель бэкенда по умолчанию")),
            Number("max_res", "Максимальная сторона", ParamType::Int, 720, 256, 8192, 1, ParamWidget::Spin, "", true),
            Tuning(Number("grid", "Сетка OFA", ParamType::Int, 1, 1, 4, 1, ParamWidget::Spin, "", true)),
            Tuning(Number("dilate", "Дилатация у границ", ParamType::Int, 1, 0, 2, 1, ParamWidget::Spin, "", true)),
            Tuning(Text("target", "Целевое разрешение mv_dlss (WxH)", "")),
            // NVDEC for OFA only: searaft / stub decode on the CPU whatever the key says (TASK-0043: it was hashed as "cuda")
            Tuning(OnlyFor(Enum("hwaccel", "Декод", "cuda", {{"cuda", "CUDA", ""}, {"none", "CPU", ""}}, ParamWidget::Select, "", true), {"ofa"})),
            Flag("fp32", "FP32", false),
            Tuning(Flag("no_dlss", "Только mv_raw", false, "не писать mv_dlss")),
            Tuning(Enum("format", "Формат пасса", "exr", Formats(false), ParamWidget::Select, "", true)),
            Text("models_dir", "Папка моделей", ""),
        };
        out.push_back(std::move(s));
    }
    {
        StageSchema s;
        s.stage = "upscale";
        s.title = "Апскейл";
        s.subtitle = "DLSS Super Resolution";
        s.guides = {"depth_dlss", "mv_dlss"};
        s.params = {
            Number("scale", "Масштаб", ParamType::Float, 2, 1.0, 4.0, 0.5, ParamWidget::Toggle, "", false, {{"1.5", "×1.5", ""}, {"2", "×2", ""}, {"3", "×3", ""}}),
            Enum("backend", "Метод", "dlss",
                 {{"dlss", "DLSS SR", "NGX, нужна nvngx_dlss.dll и RTX"},
                  {"nis", "NIS (без GPU)", "NVIDIA Image Scaling, любой D3D12"},
                  {"bicubic", "бикубик", "эталон без нейросети"},
                  {"trt", "Нейросеть (TensorRT)", "Real-ESRGAN и другие ONNX-модели из models/registry.json"},
                  {"worker", "RealBasicVSR (PyTorch)", "видео-SR с распространением по кадрам; нужна models/export/.venv"}}),
            OnlyFor(Enum("model", "Модель", "realesrgan-x2plus",
                         {{"realesrgan-x2plus", "Real-ESRGAN ×2 — качество", "RRDBNet, 23 блока, BSD-3"},
                          {"realesr-general-x4v3", "Real-ESRGAN general ×4 — быстро", "компактная сеть, ×4 затем к цели"},
                          {"realesrgan-x4plus", "Real-ESRGAN ×4 — качество", "RRDBNet ×4: для ×3 и 4K"}},
                         ParamWidget::Select, "нейросеть метода «Нейросеть (TensorRT)»"),
                    {"trt"}),
            OnlyFor(Number("tile", "Тайл", ParamType::Int, 0, 0, 1024, 64, ParamWidget::Spin, "окно нейросети в px, 0 = как в реестре (512)", true), {"trt"}),
            OnlyFor(Text("models_dir", "Папка моделей", ""), {"trt", "worker"}),
            Tuning(OnlyFor(Number("window", "Окно кадров", ParamType::Int, 0, 0, 60, 1, ParamWidget::Spin, "кадров за проход RealBasicVSR, 0 = как в реестре (15)", true),
                           {"worker"})),
            Tuning(OnlyFor(Number("overlap", "Перекрытие окон", ParamType::Int, 0, 0, 10, 1, ParamWidget::Spin,
                                  "кадров, пересчитываемых следующим окном, 0 = как в реестре (3)", true),
                           {"worker"})),
            Tuning(Number("sharpness", "Резкость NIS", ParamType::Float, 0.5, 0.0, 1.0, 0.05, ParamWidget::Slider, "", true)),
            Tuning(Text("preset", "Пресет DLSS", "default")),
            Tuning(Flag("jitter", "Джиттер DLSS", true)),
            Tuning(Number("jitter_sign", "Знак джиттера", ParamType::Float, 1.0, -1.0, 1.0, 2.0, ParamWidget::Spin, "", true)),
            Tuning(Flag("artifact_reduction_only", "Без масштабирования (×1)", false,
                        "DLSS — сглаживание DLAA, NIS — повышение резкости; артефакты сжатия не подавляются")),
            Tuning(Number("target_width", "Ширина цели", ParamType::Int, 0, 0, 7680, 2, ParamWidget::Spin, "0 = по масштабу", true)),
            Tuning(Number("target_height", "Высота цели", ParamType::Int, 0, 0, 4320, 2, ParamWidget::Spin, "0 = по масштабу", true)),
            Tuning(Flag("no_fallback", "Без запасного NIS", false, "ошибка вместо перехода на NIS")),
            Tuning(Enum("format", "Формат пасса", "exr", Formats(true), ParamWidget::Select, "", true)),
        };
        out.push_back(std::move(s));
    }
    {
        StageSchema s;
        s.stage = "nr";
        s.title = "Улучшение";
        s.subtitle = "DLSS Neural Rendering";
        s.guides = {"depth_dlss", "mv_dlss"};
        using Op = ParamCondition::Op;
        // TASK-0046: every setting of the model and of our processing around it in the card (Output/ref-review; the
        // reference Visual Enhancer shows eight). PR-2 (E7, TASK-0052): styles 3..6 gave the cinematic picture bit for bit and the
        // render preset 0..3 changed nothing on DLL 310.8.0.0 - both left the schema (the user's decision, no compatibility kept)
        ParamSpec style = Enum("style", "Стиль", "natural",
                               {{"natural", "Естественный (1)", "мягче, ближе к исходным тонам кожи и балансу"},
                                {"cinematic", "Кинематографичный (2)", "приглушает блеск и цвет"},
                                {"default", "Стандартный (0)", "сильнее всего добавляет деталь"}},
                               ParamWidget::Select,
                               "Профиль обработки модели (DLSSNR.Style). NVIDIA стили не документирует; номер в скобках совпадает со «Style N» в "
                               "Visual Enhancer. Сравните на превью.");
        ParamSpec skin = Number("skin_structure", "Структура кожи", ParamType::Float, -1.0, -1.0, 2.0, 0.05, ParamWidget::Slider,
                                "Своя сила структуры на коже, найденной автомаской. Без флажка «отдельно» — как «Локальная структура». "
                                "Действует только с автомаской.");
        skin.sentinel = -1.0;
        skin.sentinelLabel = "как структура";
        skin.sentinelFollows = "local_structure";
        skin.rangeMin = 0.0;
        s.params = {
            In(style, "Модель"),
            // PR-7 (PR-2 E1, the user's decision): the edit strength after the model - NGX's own DLSSNR.Intensity is not one (above 1 = 1,
            // below 1 darker, 0 black), the model always gets 1 and NrCompose blends
            In(Display(Number("intensity", "Интенсивность", ParamType::Float, 1.0, 0.0, 2.0, 0.05, ParamWidget::Slider,
                              "Сила правки после модели: 0 — кадр после апскейла (модель не вызывается), 1 — картинка модели, выше 1 — "
                              "усиление (экстраполяция): ярче, возможны пересветы и шум."),
                       "без модели"),
               "Модель"),
            In(Number("passes", "Проходы", ParamType::Int, 1, 1, 4, 1, ParamWidget::Toggle,
                      "Модель повторно обрабатывает собственный результат: 1 — как обучена, 2 — богаче, 3–4 — заметно синтетичнее. Каждый проход — "
                      "своя копия модели: добавляет её время и видеопамять (≈ 0.5 ГБ на 1440p, 0.9 ГБ на 4K).", false,
                      {{"1", "1", ""}, {"2", "2", ""}, {"3", "3", ""}, {"4", "4", ""}}),
               "Модель"),
            In(Number("local_tone", "Локальный тон", ParamType::Float, 1.0, 0.0, 2.0, 0.05, ParamWidget::Slider,
                      "Сила локальной правки света и тона: 0 — без неё, 1 — как обучена, до 2 — сильнее (в Visual Enhancer потолок 1)."),
               "Локальные"),
            In(Number("local_structure", "Локальная структура", ParamType::Float, 1.0, 0.0, 2.0, 0.05, ParamWidget::Slider,
                      "Сила микродеталей и локального контраста: 0 — без них, 1 — как обучена, до 2 — сильнее."),
               "Локальные"),
            In(Display(Number("temporal", "Подавление мерцания", ParamType::Float, 0.0, 0.0, 1.0, 0.05, ParamWidget::Slider,
                              "Смешивает кадр с предыдущим, совмещённым по векторам движения: убирает «дыхание» на статичных планах, на быстром "
                              "движении возможен шлейф."),
                       "выкл"),
               "Временная стабильность"),
            In(ShowIf(Number("temporal_threshold", "Порог подавления мерцания", ParamType::Float, 0.1, 0.0, 1.0, 0.01, ParamWidget::Spin,
                             "При какой разнице цвета смешивание с прошлым кадром отменяется: ниже порог — меньше шлейфов, но слабее подавление."),
                      "temporal", Op::Gt, 0.0),
               "Временная стабильность", true),
            In(Flag("guides", "Глубина и векторы", true,
                    "Подавать модели глубину и векторы движения. Без них каждый кадр обрабатывается отдельно: «дыхания» нет, но на движении "
                    "ниже согласованность во времени.",
                    false),
               "Временная стабильность", true),
            // PR-8: the control mask replaces the automatic one (UseAutoMask 0) - both rows disabled with a note while it is an input
            In(DisabledByInputs(Flag("auto_mask", "Автомаска кожи", false,
                                     "Модель сама находит кожу и применяет к ней «Структуру кожи». Пока загружена маска управления, автомаска "
                                     "выключена.",
                                     false),
                                {kControlMaskPass}, "выключена: её заменяет маска управления"),
               "Маска"),
            In(DisabledByInputs(EnableIf(skin, "auto_mask", Op::Eq, true, "действует только с автомаской"), {kControlMaskPass},
                                "не действует при маске управления"),
               "Маска"),
            In(NeedsInputs(Number("skin_blend", "Сила на коже (маски)", ParamType::Float, 1.0, 0.0, 1.0, 0.05, ParamWidget::Slider,
                                  "Доля правки внутри масок лица и кожи (импортированные пассы mask_face / mask_skin): 0 — исходник, 1 — полная правка."),
                           {"mask_face", "mask_skin"}),
               "Маска"),
            In(Enum("tonemap", "Тонмаппинг", "passthrough", {{"passthrough", "без изменений", ""}, {"aces", "ACES", ""}, {"reinhard", "Reinhard", ""}},
                    ParamWidget::Select, "Кривая перед моделью: модель ждёт SDR sRGB. Для обычного видео — «без изменений»."),
               "Вход", true),
            In(Enum("input_transfer", "Входная кривая", "srgb", {{"srgb", "sRGB", ""}, {"linear", "линейная", ""}, {"pq", "PQ", ""}, {"hlg", "HLG", ""}},
                    ParamWidget::Select, "Как закодирован вход: sRGB — обычное видео; линейная; PQ/HLG — HDR. Результат всегда SDR."),
               "Вход", true),
            // two decimals: the v1 migration stores log2 of the old multiplier at 0.01 EV (1.5 -> 0.58)
            In(Decimals(Display(Number("exposure", "Экспозиция", ParamType::Float, 0.0, -8.0, 8.0, 0.1, ParamWidget::Spin,
                                       "Сдвиг экспозиции перед тонмаппингом, в ступенях: +1 — вдвое светлее, −1 — вдвое темнее, 0 — без изменений."),
                                "", " EV"),
                        2),
               "Вход", true),
            In(Number("model_scale", "Разрешение модели", ParamType::Float, 1.0, 0.25, 2.0, 0.25, ParamWidget::Spin,
                      "Модель работает на уменьшенной (< 1) или увеличенной (> 1, суперсэмплинг) копии кадра, и правка переносится на полный кадр. "
                      "Меньше 1 — немного быстрее и заметно меняется характер правки; больше 1 — медленнее."),
               "Разрешение модели", true),
            In(ShowIf(Number("transfer", "Сила переноса", ParamType::Float, 1.0, 0.0, 1.0, 0.05, ParamWidget::Slider,
                             "Только при разрешении модели ≠ 1: какая доля правки переносится на полный кадр."),
                      "model_scale", Op::Ne, 1.0),
               "Разрешение модели", true),
            In(ShowIf(Number("max_ratio", "Предел отношения", ParamType::Float, 4.0, 1.0, 16.0, 0.5, ParamWidget::Spin,
                             "Ограничивает поканальное отношение при переносе, защищает от выбросов в тенях."),
                      "model_scale", Op::Ne, 1.0),
               "Разрешение модели", true),
            // engineer keys: a card preset never changes them (spec §2h), a pipeline preset does
            Tuning(Enum("backend", "Бэкенд", "ngx", {{"ngx", "DLSS NR (NGX)", "nvngx_dlssnr.dll"}, {"stub", "stub (тест)", ""}}, ParamWidget::Select, "", true)),
            Tuning(Enum("format", "Формат пасса", "exr", Formats(true), ParamWidget::Select, "", true)),
        };
        s.cardPresets = true;  // PR-9: the card's «Пресет ▾»
        // PR-8: the control mask - a pass, no parameter; its row after «Структура кожи» (E10: G / B are the low / high frequencies)
        s.passRows = {{kControlMaskPass, "Маска управления",
                       "Изображение-маска поверх правки: белый — правка как в карточке, чёрный — без правки. Серое изображение ослабляет правку "
                       "ровно в v раз. В цветной маске R — общая сила, G — низкие частоты правки (крупные перепады света и тона), B — высокие "
                       "(мелкая деталь); каналы перемножаются, поэтому цветной серый (v, v, v) ослабляет правку в v² раз. «Локальный тон» почти весь "
                       "в низких частотах, «Локальная структура» — в обеих. Где правки нет, нет и подавления мерцания. Одна маска на все кадры, "
                       "растягивается на кадр без сохранения пропорций; альфа не учитывается. Пока маска загружена, автомаска кожи выключена.",
                       "skin_structure"}};
        out.push_back(std::move(s));
    }
    {
        StageSchema s;
        s.stage = "fg";
        s.title = "Генерация кадров";
        s.subtitle = "DLSS Frame Generation";
        s.guides = {"depth_dlss", "mv_dlss"};
        s.params = {
            Number("multiplier", "Множитель", ParamType::Int, 2, 2, 4, 1, ParamWidget::Toggle, "×3 и ×4 — RIFE (RTX 40 даёт только ×2)", false,
                   {{"2", "×2", ""}, {"3", "×3", ""}, {"4", "×4", ""}}),
            Enum("backend", "Метод", "dlssg",
                 {{"dlssg", "DLSS Frame Generation", "nvngx_dlssg.dll, ×2 на RTX 40"}, {"rife", "RIFE (TensorRT)", "×2..×4, любой RTX"}, {"blend", "смешивание (тест)", ""}}),
            Tuning(Text("model", "Модель RIFE", "rife49")),
            Flag("fp32", "FP32", false),
            Tuning(Enum("backbuffer_format", "Формат буфера", "rgba16f", {{"rgba16f", "RGBA16F", ""}, {"rgba8", "RGBA8", ""}}, ParamWidget::Select, "", true)),
            Tuning(Enum("format", "Формат пасса", "exr", Formats(true), ParamWidget::Select, "", true)),
        };
        out.push_back(std::move(s));
    }
    return out;
}

bool IsChoice(const ParamSpec& spec, const J& v) {
    for (const auto& c : spec.choices) {
        if (v.is_string() && v.get<std::string>() == c.value) return true;
        if (v.is_number()) {
            try {
                if (std::fabs(v.get<double>() - std::stod(c.value)) < 1e-9) return true;
            } catch (...) {
            }
        }
    }
    return false;
}

std::string Brief(const J& v) { return v.is_string() ? v.get<std::string>() : v.dump(); }

}  // namespace

const ParamSpec* StageSchema::Find(const std::string& key) const {
    for (const auto& p : params)
        if (p.key == key) return &p;
    return nullptr;
}

ParamLevel LevelOf(const ParamSpec& spec) { return spec.advanced ? ParamLevel::Engineer : spec.more ? ParamLevel::More : ParamLevel::Basic; }

const std::vector<StageSchema>& StageSchemas() {
    static const std::vector<StageSchema> schemas = Build();
    return schemas;
}

const StageSchema* FindStageSchema(const std::string& stage) {
    for (const auto& s : StageSchemas())
        if (s.stage == stage) return &s;
    return nullptr;
}

nlohmann::json NormalizeStageParams(const std::string& stage, const nlohmann::json& params) {
    if (!params.is_object()) return params;
    nlohmann::json out = params;
    auto alias = [&](const char* from, const char* to) {
        if (!out.contains(from)) return;
        if (!out.contains(to)) out[to] = out[from];
        out.erase(from);
    };
    if (stage == "nr") {
        // every snake key of the schema under its kebab spelling (ApplyNrParams reads both): from the schema, so a new key cannot be
        // forgotten here - scene-cut was (review of PR-6)
        for (const ParamSpec& s : FindStageSchema("nr")->params) {
            if (s.key.find('_') == std::string::npos) continue;
            std::string kebab = s.key;
            std::replace(kebab.begin(), kebab.end(), '_', '-');
            alias(kebab.c_str(), s.key.c_str());
        }
        // a style given as a number (JSON, --param nr.style=4) or as the index of a named one: the schema's string
        if (out.contains("style") && out["style"].is_number_integer()) out["style"] = std::to_string(out["style"].get<int64_t>());
        if (out.contains("style") && out["style"].is_string()) {
            static const char* named[] = {"default", "natural", "cinematic"};
            const std::string v = out["style"].get<std::string>();
            if (v == "0" || v == "1" || v == "2") out["style"] = named[v[0] - '0'];
        }
        for (const char* inverse : {"no_guides", "no-guides"}) {
            if (!out.contains(inverse)) continue;
            // a non-boolean stays wrong under `guides` for the validation to say so
            if (!out.contains("guides")) out["guides"] = out[inverse].is_boolean() ? nlohmann::json(!out[inverse].get<bool>()) : out[inverse];
            out.erase(inverse);
        }
    } else if (stage == "fg") {
        alias("backbuffer-format", "backbuffer_format");
    }
    return out;
}

const MovedParam* FindMovedParam(const std::string& stage, const std::string& key) {
    static const MovedParam moved[] = {{"nr", "scene_cut", "scenes.threshold", "Сброс на склейках"}};  // PR-13: the clip's list
    std::string snake = key;
    std::replace(snake.begin(), snake.end(), '-', '_');
    for (const MovedParam& m : moved)
        if (stage == m.stage && snake == m.key) return &m;
    return nullptr;
}

std::string MovedParamProblem(const MovedParam& m) {
    return std::string(m.stage) + "." + m.key + " moved to " + m.to + " (PR-13): use --param " + m.to + "=... or the project's stage scenes";
}

std::vector<ParamProblem> CheckStageParams(const std::string& stage, const nlohmann::json& rawParams) {
    using Kind = ParamProblem::Kind;
    std::vector<ParamProblem> problems;
    const auto add = [&](Kind kind, const std::string& key, const J& got, std::string text) { problems.push_back({kind, stage, key, got, std::move(text)}); };
    const nlohmann::json params = NormalizeStageParams(stage, rawParams);
    const StageSchema* schema = FindStageSchema(stage);
    if (!schema) {
        add(Kind::Shape, "", J(), "unknown stage '" + stage + "'");
        return problems;
    }
    if (params.is_null()) return problems;
    if (!params.is_object()) {
        add(Kind::Shape, "", params, stage + ": parameters must be a JSON object");
        return problems;
    }
    for (const auto& [key, v] : params.items()) {
        const ParamSpec* spec = schema->Find(key);
        if (!spec) {
            if (const MovedParam* m = FindMovedParam(stage, key)) add(Kind::Moved, key, v, MovedParamProblem(*m));  // else hashed and ignored
            continue;  // backends take extra keys
        }
        const std::string where = stage + "." + key;
        switch (spec->type) {
            case ParamType::Bool:
                if (!v.is_boolean()) add(Kind::Type, key, v, where + " must be true or false (got " + Brief(v) + ")");
                break;
            case ParamType::Int:
            case ParamType::Float: {
                if (!v.is_number() || (spec->type == ParamType::Int && !v.is_number_integer() && v.get<double>() != std::floor(v.get<double>()))) {
                    add(Kind::Type, key, v, where + " must be a number (got " + Brief(v) + ")");
                    break;
                }
                if (!spec->choices.empty() && !IsChoice(*spec, v)) {
                    std::string all;
                    for (const auto& c : spec->choices) all += (all.empty() ? "" : " | ") + c.value;
                    add(Kind::Choice, key, v, where + " must be one of " + all + " (got " + Brief(v) + ")");
                } else if (spec->sentinel && v.get<double>() == *spec->sentinel) {
                    // the special value (skin_structure -1 = «as the local structure»)
                } else if (spec->max > spec->min) {
                    const double lo = spec->sentinel ? spec->rangeMin : spec->min;
                    if (v.get<double>() < lo || v.get<double>() > spec->max)
                        add(Kind::Range, key, v,
                            where + " must be " + (spec->sentinel ? J(*spec->sentinel).dump() + " or " : std::string()) + "within " + J(lo).dump() + ".." +
                                J(spec->max).dump() + " (got " + Brief(v) + ")");
                }
                break;
            }
            case ParamType::Enum:
                if (!v.is_string()) {
                    add(Kind::Type, key, v, where + " must be a string (got " + Brief(v) + ")");
                } else if (!IsChoice(*spec, v)) {
                    std::string all;
                    for (const auto& c : spec->choices) all += (all.empty() ? "" : " | ") + c.value;
                    add(Kind::Choice, key, v, where + " must be one of " + all + " (got " + Brief(v) + ")");
                }
                break;
            case ParamType::String:
                if (!v.is_string()) add(Kind::Type, key, v, where + " must be a string (got " + Brief(v) + ")");
                break;
        }
    }
    return problems;
}

std::vector<std::string> ValidateStageParams(const std::string& stage, const nlohmann::json& params) {
    std::vector<std::string> texts;
    for (ParamProblem& p : CheckStageParams(stage, params)) texts.push_back(std::move(p.text));
    return texts;
}

nlohmann::json EffectiveStageParams(const std::string& stage, const nlohmann::json& params) {
    nlohmann::json out = params.is_object() ? NormalizeStageParams(stage, params) : nlohmann::json::object();
    if (const StageSchema* schema = FindStageSchema(stage)) {
        std::string backend;
        if (const ParamSpec* b = schema->Find("backend")) {
            if (out.contains("backend") && out["backend"].is_string()) backend = out["backend"].get<std::string>();
            else if (b->def.is_string()) backend = b->def.get<std::string>();
        }
        for (const auto& p : schema->params) {
            if (!p.backends.empty() && std::find(p.backends.begin(), p.backends.end(), backend) == p.backends.end()) {
                out.erase(p.key);  // another backend's parameter: the same run whatever it says
                continue;
            }
            if (p.omitDefault) {  // absent and the default are one run, and neither enters the fingerprint
                if (out.contains(p.key) && out[p.key] == p.def) out.erase(p.key);
                continue;
            }
            if (!out.contains(p.key)) out[p.key] = p.def;
        }
    }
    return out;
}

std::string ParamValueLabel(const ParamSpec& spec, const nlohmann::json& value) {
    for (const auto& c : spec.choices) {
        ParamSpec one = spec;
        one.choices = {c};
        if (IsChoice(one, value)) return c.label;
    }
    if (value.is_boolean()) return value.get<bool>() ? "да" : "нет";
    if (value.is_number()) {
        const double d = value.get<double>();
        // the words of the form (TASK-0046): «как структура» for skin_structure -1, «выкл» for temporal 0, « EV»
        if (spec.sentinel && d == *spec.sentinel && !spec.sentinelLabel.empty()) return spec.sentinelLabel;
        if (!spec.zeroLabel.empty() && d == spec.min) return spec.zeroLabel;
        if (d == std::floor(d) && std::fabs(d) < 1e9) return std::to_string(static_cast<long long>(d)) + spec.suffix;
        char buf[32];
        std::snprintf(buf, sizeof buf, "%g", d);
        return buf + spec.suffix;
    }
    return value.is_string() ? value.get<std::string>() : value.dump();
}

std::string ParamValueLabel(const ParamSpec& spec, const nlohmann::json& value, const std::string& backend) {
    if (value.is_null() && spec.omitDefault) return "по умолчанию";
    if (const auto* list = FormChoicesFor(spec, backend))
        for (const auto& c : *list)
            if (value.is_string() && value.get<std::string>() == c.value) return c.label;
    return ParamValueLabel(spec, value);
}

namespace {

bool Holds(const ParamCondition& c, const nlohmann::json& params) {
    const nlohmann::json v = params.is_object() && params.contains(c.key) ? params[c.key] : nlohmann::json();
    switch (c.op) {
        case ParamCondition::Op::Eq:
            return v.is_number() && c.value.is_number() ? v.get<double>() == c.value.get<double>() : v == c.value;
        case ParamCondition::Op::Ne:
            return v.is_number() && c.value.is_number() ? v.get<double>() != c.value.get<double>() : v != c.value;
        case ParamCondition::Op::Gt:
            return v.is_number() && c.value.is_number() && v.get<double>() > c.value.get<double>();
    }
    return false;
}

}  // namespace

ParamRowState RowState(const ParamSpec& spec, const nlohmann::json& params, const std::set<std::string>& inputs) {
    ParamRowState st;
    for (const auto& c : spec.showIf) st.visible = st.visible && Holds(c, params);
    if (!spec.needsInputs.empty()) {
        bool any = false;
        for (const auto& in : spec.needsInputs) any = any || inputs.count(in) > 0;
        st.visible = st.visible && any;
    }
    for (const auto& in : spec.disabledByInputs)
        if (inputs.count(in)) {
            st.enabled = false;
            st.note = spec.disabledByInputsNote;
        }
    if (st.enabled)
        for (const auto& c : spec.enableIf)
            if (!Holds(c, params)) {
                st.enabled = false;
                st.note = spec.disabledNote;
            }
    return st;
}

int ParamDecimals(const ParamSpec& spec) {
    if (spec.decimals >= 0) return spec.decimals;
    if (spec.type == ParamType::Int) return 0;
    const double step = spec.step > 0 ? spec.step : 0.1;
    int d = 0;
    for (double x = step; d < 6 && std::fabs(x - std::round(x)) > 1e-9; x *= 10) ++d;
    return d;
}

std::string StageBackend(const std::string& stage, const nlohmann::json& params) {
    if (params.is_object() && params.contains("backend") && params["backend"].is_string()) return params["backend"].get<std::string>();
    if (const StageSchema* schema = FindStageSchema(stage))
        if (const ParamSpec* b = schema->Find("backend"); b && b->def.is_string()) return b->def.get<std::string>();
    return {};
}

const std::vector<ParamChoice>* FormChoicesFor(const ParamSpec& spec, const std::string& backend) {
    const auto it = spec.formChoices.find(backend);
    return it == spec.formChoices.end() ? nullptr : &it->second;
}

nlohmann::json ParamsForBackendChange(const std::string& stage, const nlohmann::json& params, const std::string& newBackend) {
    nlohmann::json out = params.is_object() ? params : nlohmann::json::object();
    const StageSchema* schema = FindStageSchema(stage);
    if (!schema) return out;
    const std::string oldBackend = StageBackend(stage, out);
    const auto inList = [](const std::vector<ParamChoice>* list, const nlohmann::json& v) {
        if (!list || !v.is_string()) return false;
        for (const auto& c : *list)
            if (c.value == v.get<std::string>()) return true;
        return false;
    };
    for (const ParamSpec& p : schema->params) {
        if (p.formChoices.empty() || !out.contains(p.key)) continue;
        const nlohmann::json& v = out[p.key];
        if (v.is_string() && v.get<std::string>().empty()) continue;  // "" = the new backend's default
        const auto* now = FormChoicesFor(p, newBackend);
        if (now ? !inList(now, v) : inList(FormChoicesFor(p, oldBackend), v)) out.erase(p.key);
    }
    return out;
}

std::vector<std::string> StageCliArgs(const std::string& stage, const nlohmann::json& params) {
    std::vector<std::string> out;
    if (!params.is_object()) return out;
    const StageSchema* schema = FindStageSchema(stage);
    const std::string backend = StageBackend(stage, params);
    int64_t targetW = 0, targetH = 0;
    for (auto it = params.begin(); it != params.end(); ++it) {
        const std::string& key = it.key();
        const nlohmann::json& v = it.value();
        const ParamSpec* spec = schema ? schema->Find(key) : nullptr;
        if (spec && !spec->backends.empty() && std::find(spec->backends.begin(), spec->backends.end(), backend) == spec->backends.end()) continue;
        if (spec && spec->omitDefault && v == spec->def) continue;
        if (stage == "upscale" && (key == "target_width" || key == "target_height") && v.is_number()) {
            (key == "target_width" ? targetW : targetH) = v.get<int64_t>();
            continue;
        }
        if (v.is_boolean()) {
            const bool def = spec && spec->def.is_boolean() && spec->def.get<bool>();
            if (v.get<bool>() == def) continue;  // the command's default
            out.push_back(v.get<bool>() ? "--" + key : "--no-" + key);
        } else if (v.is_string()) {
            if (v.get<std::string>().empty()) continue;  // «пусто = по умолчанию»
            out.push_back("--" + key);
            out.push_back(v.get<std::string>());
        } else if (!v.is_null()) {
            out.push_back("--" + key);
            out.push_back(v.dump());
        }
    }
    if (targetW > 0 && targetH > 0) {
        out.push_back("--target");
        out.push_back(std::to_string(targetW) + "x" + std::to_string(targetH));
    }
    return out;
}

}  // namespace dlssvid
