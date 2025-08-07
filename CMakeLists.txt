# CMakeLists.txt
#
# This file manages the build process for the AI Inference Acceleration project.
# It finds necessary libraries like CUDA and TensorRT and defines the executables.

cmake_minimum_required(VERSION 3.18)
project(AI_Inference_Acceleration CXX)

# --- Settings ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set the output directory for executables to ${PROJECT_SOURCE_DIR}/bin
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

# --- Find Dependencies ---

# Find CUDA and add its libraries
find_package(CUDA REQUIRED)
if(NOT CUDA_FOUND)
    message(FATAL_ERROR "CUDA Toolkit not found!")
else()
    message(STATUS "Found CUDA: ${CUDA_TOOLKIT_ROOT_DIR}")
endif()

# Define the paths for TensorRT.
# In a standard installation (like on Colab), these are the typical paths.
set(TENSORRT_INCLUDE_DIRS /usr/include/x86_64-linux-gnu)
set(TENSORRT_LIBRARY_DIRS /usr/lib/x86_64-linux-gnu)

# Check if the directories exist
if(NOT EXISTS ${TENSORRT_INCLUDE_DIRS} OR NOT EXISTS ${TENSORRT_LIBRARY_DIRS})
    message(FATAL_ERROR "TensorRT directories not found. Please check paths.")
else()
    message(STATUS "Found TensorRT include directory: ${TENSORRT_INCLUDE_DIRS}")
    message(STATUS "Found TensorRT library directory: ${TENSORRT_LIBRARY_DIRS}")
endif()

# Add the include directories for the compiler
include_directories(${CUDA_INCLUDE_DIRS} ${TENSORRT_INCLUDE_DIRS})

# --- Define Executables ---

# Target 1: The simple test program
add_executable(simple_trt_test src/01_simple_trt_test.cpp)
target_link_libraries(simple_trt_test
    PRIVATE
    ${CUDA_cudart_LIBRARY}
    ${TENSORRT_LIBRARY_DIRS}/libnvinfer.so
)
message(STATUS "Added target: simple_trt_test")

# Target 2: The ONNX to TensorRT converter
add_executable(onnx_to_trt src/02_onnx_to_trt.cpp)
target_link_libraries(onnx_to_trt
    PRIVATE
    ${CUDA_cudart_LIBRARY}
    ${TENSORRT_LIBRARY_DIRS}/libnvinfer.so
    ${TENSORRT_LIBRARY_DIRS}/libnvonnxparser.so
)
message(STATUS "Added target: onnx_to_trt")

# Target 3: The inference runner
add_executable(trt_inference src/03_trt_inference.cpp)
target_link_libraries(trt_inference
    PRIVATE
    ${CUDA_cudart_LIBRARY}
    ${TENSORRT_LIBRARY_DIRS}/libnvinfer.so
)
message(STATUS "Added target: trt_inference")

# --- Final Message ---
message(STATUS "CMake configuration complete. Run 'make' in the build directory to compile.")
