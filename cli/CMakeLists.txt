add_executable(dlssvid main.cpp ViewportCommands.h ViewportCommands.cpp UpscaleCommands.h UpscaleCommands.cpp NrCommands.h NrCommands.cpp)
target_link_libraries(dlssvid PRIVATE dlssvid_core CLI11::CLI11)
set_target_properties(dlssvid PROPERTIES OUTPUT_NAME dlssvid)

# Development convenience: the DLSS SDK's nvngx_dlss.dll next to the executables (bin/nvidia/, docs/dll-setup.md).
if(DLSSVID_WITH_DLSS AND EXISTS "${DLSS_SDK_ROOT}/lib/Windows_x86_64/rel/nvngx_dlss.dll")
  add_custom_command(TARGET dlssvid POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${DLSS_SDK_ROOT}/lib/Windows_x86_64/rel/nvngx_dlss.dll" "$<TARGET_FILE_DIR:dlssvid>/nvidia/nvngx_dlss.dll"
    VERBATIM)
endif()
