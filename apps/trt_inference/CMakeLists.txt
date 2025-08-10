add_executable(trt_inference ${CMAKE_CURRENT_SOURCE_DIR}/src/trt_inference.cpp)

target_link_libraries(trt_inference PRIVATE trt_utils)
