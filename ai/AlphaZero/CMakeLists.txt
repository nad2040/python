cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(FiveInARow LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Silence kineto warning
set(TORCH_USE_KINETO OFF)

find_package(Torch REQUIRED)

add_executable(FiveInARow FiveInARow.cpp)

# THIS is what actually adds include paths + link flags
target_link_libraries(FiveInARow PRIVATE torch)

# macOS rpath fix (important for runtime)
set_target_properties(FiveInARow PROPERTIES
    BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib"
)

# Windows DLL handling (unchanged)
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET FiveInARow
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${TORCH_DLLS}
    $<TARGET_FILE_DIR:FiveInARow>)
endif()

