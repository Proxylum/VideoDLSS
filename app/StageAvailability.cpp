#include "StageAvailability.h"

#include <QCoreApplication>

namespace dlssvid {

namespace {
QString Tr(const char* s) { return QCoreApplication::translate("StageAvailability", s); }

StageAvailability Skipped(const QString& why, const QString& tip, bool checkable = false) {
    StageAvailability a;
    a.text = Tr("%1 — стадия будет пропущена").arg(why);
    a.tone = "danger";
    a.tip = tip;
    a.checkable = checkable;
    return a;
}
}  // namespace

bool StageNeedsNgx(const std::string& stage, const nlohmann::json& params) {
    const std::string backend = params.is_object() ? params.value("backend", std::string()) : std::string();
    if (stage == "nr") return backend == "ngx" && !NrBypassParams(params);
    if (stage == "fg") return backend == "dlssg";
    return false;
}

StageAvailability StageAvailabilityFor(const std::string& stage, bool enabled, const nlohmann::json& params, const AvailabilityFacts& f, const StageCheck* check,
                                       bool checking) {
    if (!enabled || !StageNeedsNgx(stage, params)) return {};
    // known without NGX: the run skips the stage for any of these (NgxNrBackend::Init, DlssgFrameGenerator::Init)
    if (f.software) return Skipped(Tr("Нужна видеокарта NVIDIA RTX, сейчас программный адаптер"), {});
    if (!f.nvidia) return Skipped(Tr("Нужна видеокарта NVIDIA RTX"), {});
    if (stage == "nr") {
        if (!f.nr.available) {
            const QString tip = QString::fromStdString(f.nr.reason);
            if (!f.nrDll) return Skipped(Tr("Нет nvngx_dlssnr.dll"), tip);
            if (!f.nrForwarder) return Skipped(Tr("Нет переходника nvngx.dll_dlssvid.dll"), tip);
            return Skipped(Tr("Neural Rendering не собран в этой версии"), tip);
        }
        // an unreadable version is no reason: the run warns and goes on (NgxNrBackend::Init)
        if (f.driver.valid && !f.driver.AtLeast(kNrMinDriverMajor, kNrMinDriverMinor))
            return Skipped(Tr("Драйвер %1, нужен %2.%3 или новее").arg(QString::fromStdString(f.driver.ToString())).arg(kNrMinDriverMajor).arg(kNrMinDriverMinor),
                           Tr("Обновите драйвер NVIDIA"));
    } else if (!f.dlssgDll) {
        return Skipped(Tr("Нет nvngx_dlssg.dll"), QString::fromStdString(f.dlssgInstruction));
    }
    // the rest only NGX knows (an unpatched DLL on RTX 20/30/40, ×3/×4 on RTX 40, the feature's creation): the check says
    StageAvailability a;
    a.checkable = true;
    a.checking = checking;
    if (checking) {
        a.text = Tr("Проверка…");
        a.tone = "none";
    } else if (check && check->ok) {
        a.text = Tr("Работает на этой видеокарте · проверено %1").arg(check->when);
        a.tone = "ok";
        a.tip = check->gpu;
    } else if (check) {
        a.text = Tr("Проверка не прошла — стадия будет пропущена");
        a.tone = "danger";
        a.tip = check->hint;
    } else {
        a.text = Tr("Работа на этой видеокарте не проверена");
        a.tone = "none";
        a.tip = Tr("«Проверить» создаёт модель на видеокарте с настройками проекта — пару секунд");
    }
    return a;
}

}  // namespace dlssvid
