cmake_minimum_required(VERSION 3.18)
project(trt_utils_consumer CXX)

find_package(trt_utils 1.0 CONFIG REQUIRED)

add_executable(hello_trt_utils main.cpp)
target_link_libraries(hello_trt_utils PRIVATE trt_utils::trt_utils)

message(STATUS "trt_utils consumer configured successfully")
