cmake_minimum_required(VERSION 3.18)
project(AI_Inference_Acceleration CXX CUDA) # Enable CUDA as a language

# --- Settings ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set the output directory for executables
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

# Option to control building example apps when used as a subproject
option(BUILD_APPS "Build example applications (executables)" ON)
option(BUILD_YOLO_ACCELERATOR "Build YOLOv8 accelerator submodule (projects/trt-yolov8-accelerator)" ON)

# --- Find Dependencies ---
# CUDA Toolkit
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    message(STATUS "Found CUDAToolkit, which provides the CUDA::* targets.")
else()
    find_package(CUDA REQUIRED)
    if(CUDA_FOUND)
        message(STATUS "Found CUDA via legacy method: ${CUDA_TOOLKIT_ROOT_DIR}")
        add_library(CUDA::cudart UNKNOWN IMPORTED)
        set_property(TARGET CUDA::cudart PROPERTY IMPORTED_LOCATION "${CUDA_CUDART_LIBRARY}")
        set_property(TARGET CUDA::cudart PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${CUDA_INCLUDE_DIRS}")
    else()
        message(FATAL_ERROR "CUDA not found by any method!")
    endif()
endif()

# TensorRT paths (adapt to your system)
set(TENSORRT_INCLUDE_DIRS /usr/include/x86_64-linux-gnu)
set(TENSORRT_LIBRARY_DIRS /usr/lib/x86_64-linux-gnu)
if(NOT EXISTS ${TENSORRT_INCLUDE_DIRS} OR NOT EXISTS ${TENSORRT_LIBRARY_DIRS})
    message(FATAL_ERROR "TensorRT directories not found. Please check paths.")
else()
    message(STATUS "Found TensorRT include directory: ${TENSORRT_INCLUDE_DIRS}")
    message(STATUS "Found TensorRT library directory: ${TENSORRT_LIBRARY_DIRS}")
endif()
include_directories(${TENSORRT_INCLUDE_DIRS})

# OpenCV
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    message(STATUS "Found OpenCV: ${OpenCV_LIBS}")
    include_directories(${OpenCV_INCLUDE_DIRS})
endif()

# --- Subdirectories ---
add_subdirectory(libs/trt_utils)

if(BUILD_APPS)
    add_subdirectory(apps/simple_trt_test)
    add_subdirectory(apps/onnx_to_trt)
    add_subdirectory(apps/trt_inference)
    add_subdirectory(apps/trt_compare)
endif()

# --- YOLOv8 Accelerator Submodule ---
if(BUILD_YOLO_ACCELERATOR)
    if(EXISTS ${PROJECT_SOURCE_DIR}/projects/trt-yolov8-accelerator/CMakeLists.txt)
        message(STATUS "Including YOLOv8 accelerator submodule build")
        add_subdirectory(projects/trt-yolov8-accelerator)
    else()
        message(WARNING "YOLOv8 accelerator submodule not present. Skipping.")
    endif()
endif()

# --- Final Message ---
message(STATUS "CMake configuration complete. Run 'make' in the build directory to compile.")
