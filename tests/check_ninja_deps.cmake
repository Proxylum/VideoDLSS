# Regression check for the stale-object incident (stages 4 and 8): with Ninja + MSVC the header dependencies come
# from cl's /showIncludes lines matched against `msvc_deps_prefix` in rules.ninja. When the bytes the localized
# compiler prints do not match what CMake stored, ninja records NO header dependencies for the object
# (`ninja -t deps` shows "#deps 0") and the object silently survives header changes.
# ninja drops includes under "Program Files" as system headers, so a translation unit that includes only <system>
# headers legitimately has 0 deps; the check therefore looks only at sources that include a project header
# (`#include "..."`).
#   cmake -DBUILD_DIR=<build dir> -DSOURCE_DIR=<repo> [-DNINJA=<ninja.exe>] -P tests/check_ninja_deps.cmake
if(NOT BUILD_DIR OR NOT SOURCE_DIR)
  message(FATAL_ERROR "BUILD_DIR and SOURCE_DIR are required")
endif()
if(NOT NINJA)
  find_program(NINJA ninja REQUIRED)
endif()
execute_process(COMMAND "${NINJA}" -C "${BUILD_DIR}" -t deps OUTPUT_VARIABLE deps RESULT_VARIABLE rc ERROR_VARIABLE err)
if(NOT rc EQUAL 0)
  message(FATAL_ERROR "ninja -t deps failed (${rc}): ${err}")
endif()
string(REGEX MATCHALL "[^\n]+\\.cpp\\.obj: #deps 0," missing "${deps}")
string(REGEX MATCHALL "\\.cpp\\.obj: #deps [0-9]+" all "${deps}")
list(LENGTH all total)
if(total EQUAL 0)
  message(FATAL_ERROR "ninja -t deps lists no C++ objects in ${BUILD_DIR}")
endif()
set(bad "")
foreach(entry IN LISTS missing)
  # <target dir>/<relative source>.cpp.obj -> the source under SOURCE_DIR
  string(REGEX REPLACE "^[^\n]*CMakeFiles/[^/]+\\.dir/" "" rel "${entry}")
  string(REGEX REPLACE "\\.obj: #deps 0,$" "" rel "${rel}")
  string(REGEX REPLACE "\\.cpp$" ".cpp" rel "${rel}")
  set(candidates "${SOURCE_DIR}/${rel}")
  # objects of subdirectory targets carry the path relative to that directory
  foreach(sub core cli app tests)
    list(APPEND candidates "${SOURCE_DIR}/${sub}/${rel}")
  endforeach()
  set(src "")
  foreach(c IN LISTS candidates)
    if(EXISTS "${c}")
      set(src "${c}")
      break()
    endif()
  endforeach()
  if(NOT src)
    continue()  # generated sources (moc, autogen): not ours to judge
  endif()
  file(STRINGS "${src}" project_includes REGEX "^[ \t]*#[ \t]*include[ \t]+\"")
  if(project_includes)
    list(APPEND bad "${rel}")
  endif()
endforeach()
list(LENGTH bad nbad)
if(nbad GREATER 0)
  string(REPLACE ";" "\n  " shown "${bad}")
  message(FATAL_ERROR "${nbad} of ${total} C++ objects include project headers but have no recorded header dependencies "
    "(msvc_deps_prefix mismatch with the localized compiler, see CMakeLists.txt):\n  ${shown}\n"
    "Fix: install the English language pack for Visual Studio (VSLANG=1033 is set by the presets), then "
    "`cmake --build --preset release --target clean` and rebuild.")
endif()
message(STATUS "ninja header dependencies recorded for every C++ object that includes project headers (${total} objects)")
