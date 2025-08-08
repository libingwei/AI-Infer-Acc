cmake_minimum_required(VERSION 3.18)
project(AI_Inference_Acceleration CXX CUDA) # Enable CUDA as a language

# --- Settings ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set the output directory for executables
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

# --- Find Dependencies ---

# Use the modern way to find the CUDA Toolkit, which defines the CUDA::* targets
# Try to find CUDA in multiple ways for better compatibility
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    message(STATUS "Found CUDAToolkit, which provides the CUDA::* targets.")
else()
    # Fallback method for older CMake versions or environments without CUDAToolkit
    find_package(CUDA REQUIRED)
    if(CUDA_FOUND)
        message(STATUS "Found CUDA via legacy method: ${CUDA_TOOLKIT_ROOT_DIR}")
        # Create an imported target to mimic CUDA::cudart
        add_library(CUDA::cudart UNKNOWN IMPORTED)
        set_property(TARGET CUDA::cudart PROPERTY IMPORTED_LOCATION "${CUDA_CUDART_LIBRARY}")
        set_property(TARGET CUDA::cudart PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${CUDA_INCLUDE_DIRS}")
    else()
        message(FATAL_ERROR "CUDA not found by any method!")
    endif()
endif()

# Define the paths for TensorRT
set(TENSORRT_INCLUDE_DIRS /usr/include/x86_64-linux-gnu)
set(TENSORRT_LIBRARY_DIRS /usr/lib/x86_64-linux-gnu)

if(NOT EXISTS ${TENSORRT_INCLUDE_DIRS} OR NOT EXISTS ${TENSORRT_LIBRARY_DIRS})
    message(FATAL_ERROR "TensorRT directories not found. Please check paths.")
else()
    message(STATUS "Found TensorRT include directory: ${TENSORRT_INCLUDE_DIRS}")
    message(STATUS "Found TensorRT library directory: ${TENSORRT_LIBRARY_DIRS}")
endif()

# Add include directories
include_directories(${TENSORRT_INCLUDE_DIRS})

# --- Define Executables ---

# Target 1: The simple test program
add_executable(simple_trt_test src/01_simple_trt_test.cpp)
target_link_libraries(simple_trt_test
    PRIVATE
    CUDA::cudart
    ${TENSORRT_LIBRARY_DIRS}/libnvinfer.so
)
message(STATUS "Added target: simple_trt_test")

# Target 2: The ONNX to TensorRT converter
add_executable(onnx_to_trt src/02_onnx_to_trt.cpp)
target_link_libraries(onnx_to_trt
    PRIVATE
    CUDA::cudart
    ${TENSORRT_LIBRARY_DIRS}/libnvinfer.so
    ${TENSORRT_LIBRARY_DIRS}/libnvonnxparser.so
)
message(STATUS "Added target: onnx_to_trt")

# Target 3: The inference runner
add_executable(trt_inference src/03_trt_inference.cpp)
target_link_libraries(trt_inference
    PRIVATE
    CUDA::cudart
    ${TENSORRT_LIBRARY_DIRS}/libnvinfer.so
)
message(STATUS "Added target: trt_inference")

# --- Final Message ---
message(STATUS "CMake configuration complete. Run 'make' in the build directory to compile.")
