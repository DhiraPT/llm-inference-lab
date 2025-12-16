# Define a library for shared utilities (header-only for now)
add_library(common_utils INTERFACE)

# Allow other targets to include headers from this directory
target_include_directories(common_utils INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
