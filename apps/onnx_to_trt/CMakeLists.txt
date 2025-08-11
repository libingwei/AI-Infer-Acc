add_executable(onnx_to_trt src/onnx_to_trt.cpp)

# Link against trt_utils and OpenCV.
# All underlying dependencies like TensorRT are now correctly propagated
# from the trt_utils target.
target_link_libraries(onnx_to_trt PRIVATE trt_utils ${OpenCV_LIBS})

