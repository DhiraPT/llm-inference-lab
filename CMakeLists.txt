cmake_minimum_required(VERSION 3.18)
project(llm-inference-lab CXX C CUDA)

# Global Settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Optimization Flags
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-O3 -march=native -Wall -Wextra)
endif()

# Common Utilities
add_subdirectory(common)

# Phases
add_subdirectory(phase-1-cpu)
# add_subdirectory(phase-2-cpu-opt) # Uncomment when ready
# add_subdirectory(phase-3-gpu-naive) 
# add_subdirectory(phase-4-gpu-opt)
