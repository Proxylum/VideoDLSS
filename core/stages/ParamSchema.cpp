#include "stages/ParamSchema.h"

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
            Enum("backend", "Модель", "da3",
                 {{"da3", "DA3 metric (точнее)", "Depth Anything 3, метрическая глубина"},
                  {"vda", "VDA small (стабильнее)", "Video Depth Anything: меньше мерцания"},
                  {"worker", "Python worker", "внешний воркер (models/export)"},
                  {"stub", "stub (тест)", "синтетическая глубина без модели"}}),
            Enum("stabilize", "Стабилизация", "auto",
                 {{"auto", "авто", "масштаб и сдвиг по окну кадров"}, {"none", "выкл", ""}, {"scale", "масштаб", ""}, {"scale_shift", "масштаб + сдвиг", ""}},
                 ParamWidget::Toggle, "временная стабилизация метрической глубины"),
            Text("model", "Модель (id реестра)", "", "пусто = модель бэкенда по умолчанию"),
            Number("input_size", "Размер входа модели", ParamType::Int, 518, 64, 4096, 14, ParamWidget::Spin, "", true),
            Number("max_res", "Максимальная сторона", ParamType::Int, 1080, 256, 8192, 1, ParamWidget::Spin, "", true),
            Number("stabilize_window", "Окно стабилизации", ParamType::Int, 8, 1, 64, 1, ParamWidget::Spin, "", true),
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
            Enum("hwaccel", "Декод", "cuda", {{"cuda", "CUDA", ""}, {"none", "CPU", ""}}, ParamWidget::Select, "", true),
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
                  {"rtxvsr", "RTX VSR", "недоступно без RTX Video SDK"}}),
            Number("sharpness", "Резкость NIS", ParamType::Float, 0.5, 0.0, 1.0, 0.05, ParamWidget::Slider, "", true),
            Text("preset", "Пресет DLSS", "default"),
            Flag("jitter", "Джиттер DLSS", true),
            Number("jitter_sign", "Знак джиттера", ParamType::Float, 1.0, -1.0, 1.0, 2.0, ParamWidget::Spin, "", true),
            Flag("artifact_reduction_only", "Только подавление артефактов", false),
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
            Number("exposure", "Экспозиция", ParamType::Float, 0.0, -8.0, 8.0, 0.1, ParamWidget::Spin, "", true),
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

std::vector<std::string> ValidateStageParams(const std::string& stage, const nlohmann::json& params) {
    std::vector<std::string> problems;
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
    nlohmann::json out = params.is_object() ? params : nlohmann::json::object();
    if (const StageSchema* schema = FindStageSchema(stage))
        for (const auto& p : schema->params)
            if (!out.contains(p.key)) out[p.key] = p.def;
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

}  // namespace dlssvid
