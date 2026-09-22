qt_standard_project_setup()

# UI library (linked by the executable and by tests/app).
qt_add_library(dlssvid_app STATIC
  AppModel.h AppModel.cpp
  SourceNames.h SourceNames.cpp
  CompareBar.h CompareBar.cpp
  StageCard.h StageCard.cpp
  StartPage.h StartPage.cpp
  ViewportWindow.h ViewportWindow.cpp
  TimelineWidget.h TimelineWidget.cpp
  ProjectPanel.h ProjectPanel.cpp
  InspectorPanel.h InspectorPanel.cpp
  LogPanel.h LogPanel.cpp
  TaskQueue.h TaskQueue.cpp
  ProcessingPanel.h ProcessingPanel.cpp
  ResultBar.h ResultBar.cpp
  MainWindow.h MainWindow.cpp
)
target_include_directories(dlssvid_app PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(dlssvid_app PUBLIC dlssvid_core Qt6::Widgets)
target_compile_definitions(dlssvid_app PUBLIC
  DLSSVID_CLI_NAME="dlssvid.exe"
  DLSSVID_CLI_PATH="$<TARGET_FILE:dlssvid>")
add_dependencies(dlssvid_app dlssvid)

qt_add_executable(dlssvid-gui WIN32 main.cpp)
target_link_libraries(dlssvid-gui PRIVATE dlssvid_app)
# the GUI runs stages through the CLI: keep dlssvid.exe next to it
add_custom_command(TARGET dlssvid-gui POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different "$<TARGET_FILE:dlssvid>" "$<TARGET_FILE_DIR:dlssvid-gui>"
  VERBATIM)
dlssvid_deploy_qt(dlssvid-gui)
