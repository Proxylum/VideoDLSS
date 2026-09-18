#include "util/Log.h"

#include <mutex>
#include <spdlog/sinks/stdout_color_sinks.h>

namespace dlssvid {

std::shared_ptr<spdlog::logger> Log() {
    static std::once_flag once;
    static std::shared_ptr<spdlog::logger> logger;
    std::call_once(once, [] {
        logger = spdlog::get("dlssvid");
        if (!logger) {
            logger = spdlog::stdout_color_mt("dlssvid");
            logger->set_pattern("[%H:%M:%S.%e] [%^%l%$] %v");
            logger->set_level(spdlog::level::info);
        }
    });
    return logger;
}

void SetLogLevel(spdlog::level::level_enum level) { Log()->set_level(level); }

}  // namespace dlssvid
