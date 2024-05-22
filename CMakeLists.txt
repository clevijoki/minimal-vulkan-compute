# CMakeLists.txt
cmake_minimum_required(VERSION 3.7)

project(minimal_vulkan_compute)

find_package(Vulkan REQUIRED)
find_package(VulkanMemoryAllocator CONFIG REQUIRED)
find_package(vk-bootstrap CONFIG REQUIRED)

add_executable(main
    source/main.cpp
    source/vma.cpp
)
target_link_libraries(main PRIVATE Vulkan::Vulkan vk-bootstrap::vk-bootstrap GPUOpen::VulkanMemoryAllocator)
set_target_properties(main PROPERTIES GLOBAL PROPERTY CXX_STANDARD 20)

if (MSVC)
    target_compile_definitions(main PRIVATE -D_CRT_SECURE_NO_WARNINGS)
endif()

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/convolution.spv
    COMMAND vcpkg_installed/x64-windows/tools/shaderc/glslc.exe ${CMAKE_CURRENT_SOURCE_DIR}/source/convolution.comp -o ${CMAKE_CURRENT_BINARY_DIR}/convolution.spv
    MAIN_DEPENDENCY ${CMAKE_CURRENT_SOURCE_DIR}/source/convolution.comp
    VERBATIM
)

add_custom_target(Shaders DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/convolution.spv)

add_dependencies(main Shaders)