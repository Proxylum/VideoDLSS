#include "stages/ParamSchema.h"

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
            Number("stabilize_window", "Окно стабилизации, ± кадров", ParamType::Int, 8, 1, 64, 1, ParamWidget::Spin,
                   "кадр выравнивается по кадрам t−N…t+N; глубина выходит с задержкой N кадров", true),
            Number("z_near", "Ближняя плоскость, м", ParamType::Float, 0.1, 0.001, 100.0, 0.01, ParamWidget::Spin, "", true),
            Number("z_far", "Дальняя плоскость, м", ParamType::Float, 1000.0, 1.0, 100000.0, 1.0, ParamWidget::Spin, "", true),
            Flag("fp32", "FP32", false, "модель в fp32 вместо fp16"),
            Flag("no_fill", "Не заполнять дыры", false),
            Flag("no_dlss", "Только depth_raw", false, "не писать depth_dlss"),
            Enum("format", "Формат пасса", "exr", Formats(false), ParamWidget::Select, "", true),
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
            Text("model", "Модель (id реестра)", "", "пусто = модель бэкенда по умолчанию"),
            Number("max_res", "Максимальная сторона", ParamType::Int, 720, 256, 8192, 1, ParamWidget::Spin, "", true),
            Number("grid", "Сетка OFA", ParamType::Int, 1, 1, 4, 1, ParamWidget::Spin, "", true),
            Number("dilate", "Дилатация у границ", ParamType::Int, 1, 0, 2, 1, ParamWidget::Spin, "", true),
            Text("target", "Целевое разрешение mv_dlss (WxH)", ""),
            // NVDEC for OFA only: searaft / stub decode on the CPU whatever the key says (TASK-0043: it was hashed as "cuda")
            OnlyFor(Enum("hwaccel", "Декод", "cuda", {{"cuda", "CUDA", ""}, {"none", "CPU", ""}}, ParamWidget::Select, "", true), {"ofa"}),
            Flag("fp32", "FP32", false),
            Flag("no_dlss", "Только mv_raw", false, "не писать mv_dlss"),
            Enum("format", "Формат пасса", "exr", Formats(false), ParamWidget::Select, "", true),
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
            OnlyFor(Number("window", "Окно кадров", ParamType::Int, 0, 0, 60, 1, ParamWidget::Spin, "кадров за проход RealBasicVSR, 0 = как в реестре (15)", true), {"worker"}),
            OnlyFor(Number("overlap", "Перекрытие окон", ParamType::Int, 0, 0, 10, 1, ParamWidget::Spin, "кадров, пересчитываемых следующим окном, 0 = как в реестре (3)", true), {"worker"}),
            Number("sharpness", "Резкость NIS", ParamType::Float, 0.5, 0.0, 1.0, 0.05, ParamWidget::Slider, "", true),
            Text("preset", "Пресет DLSS", "default"),
            Flag("jitter", "Джиттер DLSS", true),
            Number("jitter_sign", "Знак джиттера", ParamType::Float, 1.0, -1.0, 1.0, 2.0, ParamWidget::Spin, "", true),
            Flag("artifact_reduction_only", "Без масштабирования (×1)", false,
                 "DLSS — сглаживание DLAA, NIS — повышение резкости; артефакты сжатия не подавляются"),
            Number("target_width", "Ширина цели", ParamType::Int, 0, 0, 7680, 2, ParamWidget::Spin, "0 = по масштабу", true),
            Number("target_height", "Высота цели", ParamType::Int, 0, 0, 4320, 2, ParamWidget::Spin, "0 = по масштабу", true),
            Flag("no_fallback", "Без запасного NIS", false, "ошибка вместо перехода на NIS"),
            Enum("format", "Формат пасса", "exr", Formats(true), ParamWidget::Select, "", true),
        };
        out.push_back(std::move(s));
    }
    {
        StageSchema s;
        s.stage = "nr";
        s.title = "Улучшение";
        s.subtitle = "DLSS Neural Rendering";
        s.guides = {"depth_dlss", "mv_dlss"};
        s.params = {
            Number("intensity", "Интенсивность", ParamType::Float, 1.0, 0.0, 2.0, 0.1, ParamWidget::Slider, "сила модели: 0 — без изменений, 2 — максимум"),
            Enum("backend", "Бэкенд", "ngx", {{"ngx", "DLSS NR (NGX)", "nvngx_dlssnr.dll"}, {"stub", "stub (тест)", ""}}, ParamWidget::Select, "", true),
            Enum("style", "Стиль", "natural", {{"natural", "естественный", ""}, {"cinematic", "кинематографичный", ""}, {"default", "по умолчанию", ""}},
                 ParamWidget::Select, "", true),
            Number("preset", "Пресет", ParamType::Int, 3, 0, 3, 1, ParamWidget::Spin, "", true),
            Number("passes", "Проходов модели", ParamType::Int, 1, 1, 2, 1, ParamWidget::Spin, "", true),
            Number("model_scale", "Масштаб модели", ParamType::Float, 1.0, 0.25, 2.0, 0.25, ParamWidget::Spin, "", true),
            Number("transfer", "Перенос отношения", ParamType::Float, 1.0, 0.0, 1.0, 0.1, ParamWidget::Spin, "", true),
            Number("max_ratio", "Предел отношения", ParamType::Float, 4.0, 1.0, 16.0, 0.5, ParamWidget::Spin, "", true),
            Number("temporal", "Временной фильтр", ParamType::Float, 0.0, 0.0, 1.0, 0.1, ParamWidget::Slider, "", true),
            Number("temporal_threshold", "Порог временного фильтра", ParamType::Float, 0.1, 0.0, 1.0, 0.05, ParamWidget::Spin, "", true),
            Number("skin_blend", "Сила на коже", ParamType::Float, 1.0, 0.0, 1.0, 0.1, ParamWidget::Slider, "", true),
            Number("local_tone", "Локальный тон", ParamType::Float, 1.0, 0.0, 2.0, 0.1, ParamWidget::Spin, "", true),
            Number("local_structure", "Локальная структура", ParamType::Float, 1.0, 0.0, 2.0, 0.1, ParamWidget::Spin, "", true),
            Number("skin_structure", "Структура кожи", ParamType::Float, -1.0, -1.0, 2.0, 0.1, ParamWidget::Spin, "-1 = как модель", true),
            Flag("auto_mask", "Автомаска", false),
            Flag("guides", "Guides (глубина и векторы)", true, "без guides — still-режим, качество ниже"),
            Enum("tonemap", "Тонмаппинг", "passthrough", {{"passthrough", "без изменений", ""}, {"aces", "ACES", ""}, {"reinhard", "Reinhard", ""}},
                 ParamWidget::Select, "", true),
            Number("exposure", "Экспозиция", ParamType::Float, 0.0, -8.0, 8.0, 0.1, ParamWidget::Spin,
                   "в ступенях EV перед тонмаппингом: +1 — вдвое светлее, −1 — вдвое темнее, 0 — без изменений", true),
            Enum("input_transfer", "Входная кривая", "srgb", {{"srgb", "sRGB", ""}, {"linear", "линейная", ""}, {"pq", "PQ", ""}, {"hlg", "HLG", ""}},
                 ParamWidget::Select, "", true),
            Enum("format", "Формат пасса", "exr", Formats(true), ParamWidget::Select, "", true),
        };
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
            Text("model", "Модель RIFE", "rife49"),
            Flag("fp32", "FP32", false),
            Enum("backbuffer_format", "Формат буфера", "rgba16f", {{"rgba16f", "RGBA16F", ""}, {"rgba8", "RGBA8", ""}}, ParamWidget::Select, "", true),
            Enum("format", "Формат пасса", "exr", Formats(true), ParamWidget::Select, "", true),
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
        for (const char* key : {"local_tone", "local_structure", "skin_structure", "auto_mask", "model_scale", "max_ratio", "temporal_threshold", "skin_blend",
                                "input_transfer"}) {
            std::string kebab = key;
            std::replace(kebab.begin(), kebab.end(), '_', '-');
            alias(kebab.c_str(), key);
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

std::vector<std::string> ValidateStageParams(const std::string& stage, const nlohmann::json& rawParams) {
    std::vector<std::string> problems;
    const nlohmann::json params = NormalizeStageParams(stage, rawParams);
    const StageSchema* schema = FindStageSchema(stage);
    if (!schema) {
        problems.push_back("unknown stage '" + stage + "'");
        return problems;
    }
    if (params.is_null()) return problems;
    if (!params.is_object()) {
        problems.push_back(stage + ": parameters must be a JSON object");
        return problems;
    }
    for (const auto& [key, v] : params.items()) {
        const ParamSpec* spec = schema->Find(key);
        if (!spec) continue;  // backends take extra keys
        const std::string where = stage + "." + key;
        switch (spec->type) {
            case ParamType::Bool:
                if (!v.is_boolean()) problems.push_back(where + " must be true or false (got " + Brief(v) + ")");
                break;
            case ParamType::Int:
            case ParamType::Float: {
                if (!v.is_number() || (spec->type == ParamType::Int && !v.is_number_integer() && v.get<double>() != std::floor(v.get<double>()))) {
                    problems.push_back(where + " must be a number (got " + Brief(v) + ")");
                    break;
                }
                if (!spec->choices.empty() && !IsChoice(*spec, v)) {
                    std::string all;
                    for (const auto& c : spec->choices) all += (all.empty() ? "" : " | ") + c.value;
                    problems.push_back(where + " must be one of " + all + " (got " + Brief(v) + ")");
                } else if (spec->max > spec->min && (v.get<double>() < spec->min || v.get<double>() > spec->max)) {
                    problems.push_back(where + " must be within " + J(spec->min).dump() + ".." + J(spec->max).dump() + " (got " + Brief(v) + ")");
                }
                break;
            }
            case ParamType::Enum:
                if (!v.is_string()) {
                    problems.push_back(where + " must be a string (got " + Brief(v) + ")");
                } else if (!IsChoice(*spec, v)) {
                    std::string all;
                    for (const auto& c : spec->choices) all += (all.empty() ? "" : " | ") + c.value;
                    problems.push_back(where + " must be one of " + all + " (got " + Brief(v) + ")");
                }
                break;
            case ParamType::String:
                if (!v.is_string()) problems.push_back(where + " must be a string (got " + Brief(v) + ")");
                break;
        }
    }
    return problems;
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
        if (d == std::floor(d) && std::fabs(d) < 1e9) return std::to_string(static_cast<long long>(d));
        char buf[32];
        std::snprintf(buf, sizeof buf, "%g", d);
        return buf;
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
