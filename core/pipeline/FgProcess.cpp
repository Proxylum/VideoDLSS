#include "pipeline/FgProcess.h"

#include <windows.h>

#include <charconv>
#include <cstdio>
#include <optional>
#include <regex>

#include "pipeline/ProcessRunner.h"
#include "stages/fg/IFrameGenerator.h"
#include "util/Log.h"
#include "util/Subprocess.h"

namespace dlssvid {

namespace {

std::string Hex(uint32_t code) {
    char buf[16];
    std::snprintf(buf, sizeof(buf), "0x%08X", code);
    return buf;
}

bool StartsWith(const std::string& s, std::string_view prefix) { return s.size() >= prefix.size() && s.compare(0, prefix.size(), prefix) == 0; }

}  // namespace

std::vector<std::string> FgProcessArgs(const FgProcessSpec& spec) {
    std::vector<std::string> a = {Utf8Arg(spec.exe)};
    if (spec.verbose) a.push_back("-v");
    // paths as the runner got them (the child inherits the working directory): making them absolute would add the working
    // directory's characters, which the child's ANSI-codepage argv may not carry where the parent's relative path did
    a.insert(a.end(), {"fg", "--as-stage", "-i", Utf8Arg(spec.input), "-o", Utf8Arg(spec.passesRoot), "--no-auto"});
    // explicit pass folders, exactly the ones the runner chose (--no-auto: the child does not pick its own)
    if (!spec.colorDir.empty()) a.insert(a.end(), {"--color-dir", Utf8Arg(spec.colorDir)});
    if (!spec.depthDir.empty()) a.insert(a.end(), {"--depth-dir", Utf8Arg(spec.depthDir)});
    if (!spec.mvDir.empty()) a.insert(a.end(), {"--mv-dir", Utf8Arg(spec.mvDir)});
    // ensure_ascii: the JSON survives the child's ANSI-codepage argv whatever the parameters hold
    a.insert(a.end(), {"--format", spec.format, "--params-json", spec.params.dump(-1, ' ', true, nlohmann::json::error_handler_t::replace)});
    if (spec.frames > 0) a.insert(a.end(), {"--frames", std::to_string(spec.frames)});
    if (spec.warp) a.push_back("--warp");
    if (spec.disableUnavailable) a.push_back("--disable-unavailable");
    if (!spec.sourceHash.empty()) a.insert(a.end(), {"--source-hash", spec.sourceHash});
    if (spec.crashAfter >= 0) a.insert(a.end(), {"--crash-after", std::to_string(spec.crashAfter)});
    if (spec.crashOnExit) a.push_back("--crash-on-exit");
    if (spec.stall) a.push_back("--stall");
    return a;
}

std::string DescribeExitCode(uint32_t code) {
    switch (code) {
        case 0xC0000005u: return Hex(code) + " (access violation)";
        case 0xC00000FDu: return Hex(code) + " (stack overflow)";
        case 0xC0000409u: return Hex(code) + " (stack buffer overrun / fast fail)";
        case 0xC0000374u: return Hex(code) + " (heap corruption)";
        case 0xC000001Du: return Hex(code) + " (illegal instruction)";
        case 0xC0000094u: return Hex(code) + " (integer divide by zero)";
        case 3u: return "3 (abort)";
        default: break;
    }
    if (code >= 0xC0000000u) return Hex(code);
    if (code >= 100 && code <= 120) return std::to_string(code) + " (command line rejected; see stderr)";  // CLI11 parse errors
    return std::to_string(code);
}

FgProcessOutcome RunFgProcess(const FgProcessSpec& spec, const std::function<void(int64_t)>& progress) {
    FgProcessOutcome out;
    Subprocess child;
    try {
        child.Start(FgProcessArgs(spec), {}, true);
    } catch (const std::exception& e) {
        out.error = std::string("cannot start the fg process: ") + e.what();
        return out;
    }
    out.pid = child.Pid();
    Log()->info("process: fg runs in its own process (pid {})", out.pid);
    child.CloseStdin();

    static const std::regex errorLine(R"(\[(?:error|critical)\]\s*(.*)$)");
    std::optional<nlohmann::json> result;
    ULONGLONG resultAt = 0;
    bool ended = false;  // ended by us: the child did not exit after its result
    std::string lastError;
    for (;;) {
        const std::optional<std::string> line = child.ReadLine(result ? 1000u : 0xFFFFFFFFu);
        if (!line) {
            if (!child.Running()) break;  // EOF: the child exited
            if (result && GetTickCount64() - resultAt > spec.exitGraceMs) {
                Log()->warn("process: the fg process did not exit {} s after its result — ending it", spec.exitGraceMs / 1000);
                child.Terminate();
                ended = true;
                break;
            }
            continue;
        }
        if (StartsWith(*line, kFgProgressPrefix)) {
            int64_t n = 0;
            const char* b = line->data() + kFgProgressPrefix.size();
            if (std::from_chars(b, line->data() + line->size(), n).ec == std::errc{}) {
                out.framesDone = n;
                if (progress) progress(n);
            }
            continue;
        }
        if (StartsWith(*line, kFgResultPrefix)) {
            try {
                result = nlohmann::json::parse(line->substr(kFgResultPrefix.size()));
                resultAt = GetTickCount64();
            } catch (const std::exception& e) {
                Log()->warn("process: unreadable result line of the fg process: {}", e.what());
            }
            continue;
        }
        std::fwrite(line->data(), 1, line->size(), stdout);  // the child's log, as is
        std::fputc('\n', stdout);
        std::fflush(stdout);
        std::smatch m;
        if (std::regex_search(*line, m, errorLine)) lastError = m[1].str();
    }
    out.exitCode = child.Wait();
    out.crashed = !ended && (out.exitCode >= 0xC0000000u || out.exitCode == 3u);
    const std::string code = DescribeExitCode(out.exitCode);

    if (result && result->is_object()) {
        out.result = *result;
        out.disabled = result->value("disabled", false);
        out.disabledReason = result->value("disabled_reason", std::string());
        if (out.disabled) {
            out.ok = true;  // nothing was written; whatever happened on exit does not matter
        } else {
            int64_t frames = 0;
            int multiplier = 0;
            if (const auto s = result->find("stats"); s != result->end() && s->is_object()) {
                frames = s->value("frames", int64_t{0});
                multiplier = s->value("multiplier", 0);
            }
            // a child that dies on exit (driver / NGX unload) after writing its result keeps the complete pass
            out.ok = out.exitCode == 0 || PassComplete(spec.passesRoot / "color_fg", FgFrameCount(frames, multiplier), spec.sourceHash);
            if (out.ok && out.exitCode != 0)
                out.warning = "the fg process " + std::string(ended ? "did not exit" : "crashed on exit") + " (exit code " + code + ") after writing its result";
            if (!out.ok) out.error = "the fg process ended with exit code " + code + " after its result, and color_fg is incomplete";
        }
        return out;
    }
    if (out.crashed)
        out.error = "the fg process crashed (exit code " + code + ") after " + std::to_string(out.framesDone) + " frame(s)";
    else if (out.exitCode == 0)
        out.error = "the fg process exited without a result";
    else
        out.error = (lastError.empty() ? std::string("the fg process failed") : lastError) + " (exit code " + code + ")";
    return out;
}

}  // namespace dlssvid
