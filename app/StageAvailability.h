#pragma once

#include <QString>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <string>

#include "stages/nr/INrBackend.h"

namespace dlssvid {

// TASK-0071: whether Neural Rendering and DLSS Frame Generation can run on this computer. `process` skips such a stage when it cannot
// (the GUI always passes --disable-unavailable) - the card says so before the run: what the GUI knows without NGX (the GPU, the driver,
// the DLL files), or what the last «Проверить» (`dlssvid nr|fg --check` with the project's settings) said.

// What the GUI process knows without NGX.
struct AvailabilityFacts {
    bool software = false;  // the WARP adapter
    bool nvidia = false;
    NvidiaDriverVersion driver;
    NrAvailability nr;        // NrAvailable("ngx"): the SDK built in, the DLL and the forwarder there
    bool nrDll = false;       // FindNrDll
    bool nrForwarder = false; // FindNrForwarder
    bool dlssgDll = false;    // FindDlssgDll
    std::string dlssgInstruction;
};

// What a check said.
struct StageCheck {
    bool ok = false;
    QString hint;  // the CLI's: why not, what to do
    QString gpu;   // «RTX 4070 Ti SUPER · драйвер 616.92»
    QString when;  // «14:02»
};

struct StageAvailability {
    QString text;  // «Нет nvngx_dlssnr.dll — стадия будет пропущена», «Работает на этой видеокарте · проверено 14:02»; empty: no line
    QString tone;  // ok | warn | danger | none
    QString tip;   // the instruction or the CLI's hint
    bool checkable = false;  // «Проверить» applies: an NVIDIA GPU, the files there
    bool checking = false;
};

// `params`: the stage's effective parameters. `check`: the last check of these settings and files, or null.
// `reused`: the plan reuses or restores the stage's pass - it goes into the result anyway, only a recompute would be skipped.
StageAvailability StageAvailabilityFor(const std::string& stage, bool enabled, const nlohmann::json& params, const AvailabilityFacts& facts, const StageCheck* check,
                                       bool checking, bool reused = false);

// The stages that have the line: nr with the ngx backend (not at intensity 0: no model then), fg with dlssg.
bool StageNeedsNgx(const std::string& stage, const nlohmann::json& params);

}  // namespace dlssvid
