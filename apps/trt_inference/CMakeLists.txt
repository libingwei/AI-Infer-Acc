add_executable(trt_inference ${CMAKE_CURRENT_SOURCE_DIR}/src/trt_inference.cpp)

# Link against trt_utils.
# All underlying dependencies like TensorRT are now correctly propagated
# from the trt_utils target.
target_link_libraries(trt_inference PRIVATE trt_utils)

