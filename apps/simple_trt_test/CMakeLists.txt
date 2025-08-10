add_executable(simple_trt_test ${CMAKE_CURRENT_SOURCE_DIR}/src/simple_trt_test.cpp)

target_link_libraries(simple_trt_test PRIVATE trt_utils)
