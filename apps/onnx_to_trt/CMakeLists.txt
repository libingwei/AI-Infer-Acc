add_executable(onnx_to_trt ${CMAKE_CURRENT_SOURCE_DIR}/src/onnx_to_trt.cpp)

target_link_libraries(onnx_to_trt PRIVATE trt_utils ${OpenCV_LIBS})
