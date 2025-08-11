add_executable(simple_trt_test ${CMAKE_CURRENT_SOURCE_DIR}/src/simple_trt_test.cpp)

# Link against trt_utils.
# All underlying dependencies like TensorRT are now correctly propagated
# from the trt_utils target, thanks to the logic in the root and trt_utils CMakeLists.
target_link_libraries(simple_trt_test PRIVATE trt_utils)



