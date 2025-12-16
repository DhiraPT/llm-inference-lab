add_executable(inference_cpu inference.cpp)

# Link common utilities
target_link_libraries(inference_cpu PRIVATE common_utils)

# Output the binary to the source folder (so README commands work)
set_target_properties(inference_cpu PROPERTIES 
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    OUTPUT_NAME "inference"
)
