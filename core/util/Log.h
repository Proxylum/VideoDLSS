#pragma once

#include <memory>
#include <spdlog/spdlog.h>

namespace dlssvid {

// Process-wide logger ("dlssvid"). Created lazily; safe to call from any thread.
std::shared_ptr<spdlog::logger> Log();

void SetLogLevel(spdlog::level::level_enum level);

}  // namespace dlssvid
