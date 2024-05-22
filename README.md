# minimal-vulkan-compute
An example program showing how to get started using compute in Vulkan. This is intended for people who might want to 

It uses:

* vkBootstrap
* VulkanMemoryAllocator
* shaderc (for glslc)

To simplify creation


# Building

## Prerequisites

* cmake
* vcpkg
* ninja and/or visual studio

## Visual Studio
`cmake --preset visual_studio`

Then open and build and run 'main'

## Ninja
```
cmake --preset debug
cmake --build build/debug
```
