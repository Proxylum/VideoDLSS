add_executable(dlssvid main.cpp ViewportCommands.h ViewportCommands.cpp UpscaleCommands.h UpscaleCommands.cpp NrCommands.h NrCommands.cpp FgCommands.h FgCommands.cpp ProcessCommands.h ProcessCommands.cpp)
target_link_libraries(dlssvid PRIVATE dlssvid_core CLI11::CLI11)
set_target_properties(dlssvid PROPERTIES OUTPUT_NAME dlssvid)

# Development convenience: the DLSS SDK's nvngx_dlss.dll next to the executables (bin/nvidia/, docs/dll-setup.md).
if(DLSSVID_WITH_DLSS AND EXISTS "${DLSS_SDK_ROOT}/lib/Windows_x86_64/rel/nvngx_dlss.dll")
  add_custom_command(TARGET dlssvid POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${DLSS_SDK_ROOT}/lib/Windows_x86_64/rel/nvngx_dlss.dll" "$<TARGET_FILE_DIR:dlssvid>/nvidia/nvngx_dlss.dll"
    VERBATIM)
endif()
# Same for DLSS Frame Generation (stage 7): nvngx_dlssg.dll next to the executables.
if(DLSSVID_WITH_DLSS AND EXISTS "${DLSS_SDK_ROOT}/lib/Windows_x86_64/rel/nvngx_dlssg.dll")
  add_custom_command(TARGET dlssvid POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${DLSS_SDK_ROOT}/lib/Windows_x86_64/rel/nvngx_dlssg.dll" "$<TARGET_FILE_DIR:dlssvid>/nvidia/nvngx_dlssg.dll"
    VERBATIM)
endif()
