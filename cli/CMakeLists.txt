add_executable(dlssvid main.cpp)
target_link_libraries(dlssvid PRIVATE dlssvid_core CLI11::CLI11)
set_target_properties(dlssvid PROPERTIES OUTPUT_NAME dlssvid)
