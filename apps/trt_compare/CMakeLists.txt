add_executable(trt_compare src/trt_compare.cpp)

# Link against trt_utils and OpenCV.
# All underlying dependencies like TensorRT are now correctly propagated
# from the trt_utils target.
target_link_libraries(trt_compare PRIVATE trt_utils ${OpenCV_LIBS})

