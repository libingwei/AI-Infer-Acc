add_executable(trt_compare ${CMAKE_CURRENT_SOURCE_DIR}/src/trt_compare.cpp)

target_link_libraries(trt_compare PRIVATE trt_utils ${OpenCV_LIBS})
