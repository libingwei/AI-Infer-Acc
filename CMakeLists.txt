cmake_minimum_required(VERSION 3.18)
project(AI_Inference_Acceleration CXX CUDA) # Enable CUDA as a language

# --- Settings ---
# This project exclusively builds shared libraries for simplicity and robust dependency management.
set(BUILD_SHARED_LIBS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set the output directory for executables and shared libraries to a central "bin"
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)

# Set RPATH so executables find .so files in their own directory (`bin`) at runtime
set(CMAKE_INSTALL_RPATH "$ORIGIN")
set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)

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

# --- Find CUDNN (robust search for versioned shared libs; headers optional) ---
# Helper: check if a library path points to a "real" .so (not a broken placeholder)
function(_is_real_lib LIBPATH OUT_VAR)
    if(NOT EXISTS "${LIBPATH}")
        set(${OUT_VAR} FALSE PARENT_SCOPE)
        return()
    endif()
    get_filename_component(_real "${LIBPATH}" REALPATH)
    if(NOT EXISTS "${_real}")
        set(${OUT_VAR} FALSE PARENT_SCOPE)
        return()
    endif()
    # Real cuDNN/TRT .so are typically large. Filter out tiny placeholder files (< 1MB).
    file(SIZE "${_real}" _sz)
    if(_sz GREATER 1000000)
        set(${OUT_VAR} TRUE PARENT_SCOPE)
    else()
        set(${OUT_VAR} FALSE PARENT_SCOPE)
    endif()
endfunction()
set(CUDNN_LIBRARY "" CACHE PATH "CUDNN library (full path to libcudnn*.so)")
set(CUDNN_INCLUDE_DIR "" CACHE PATH "CUDNN include directory (optional)")
set(TRT_CUDNN_LIBS "" CACHE STRING "List of cuDNN libraries to link (auto-detected)")

# Collect candidate directories to search in
set(_CUDNN_CANDIDATE_DIRS)
if(CUDAToolkit_ROOT)
    list(APPEND _CUDNN_CANDIDATE_DIRS
        ${CUDAToolkit_ROOT}/lib
        ${CUDAToolkit_ROOT}/lib64
        ${CUDAToolkit_ROOT}/targets/x86_64-linux-gnu/lib
    )
endif()
list(APPEND _CUDNN_CANDIDATE_DIRS
    /usr/lib/x86_64-linux-gnu
    /usr/local/cuda/lib64
    /usr/local/cuda/targets/x86_64-linux-gnu/lib
    /usr/lib
)

# Kaggle datasets: also search under /kaggle/input/* if present
if(EXISTS "/kaggle/input")
    file(GLOB _KAGGLE_ROOTS "/kaggle/input/*")
    foreach(_kr ${_KAGGLE_ROOTS})
        list(APPEND _CUDNN_CANDIDATE_DIRS
            ${_kr}/lib
            ${_kr}/lib64
            ${_kr}/targets/x86_64-linux-gnu/lib
        )
    endforeach()
endif()

# Try to find the primary cuDNN shared library (prefer real, versioned files)
if(NOT CUDNN_LIBRARY)
    foreach(_d ${_CUDNN_CANDIDATE_DIRS})
        if(EXISTS "${_d}")
            # Prefer base libcudnn*.so, then fall back to infer component libs
            # Strictly prefer cuDNN v8 to match TensorRT 8.x
            file(GLOB _candidates_base "${_d}/libcudnn.so.8*")
            file(GLOB _candidates_infer "${_d}/libcudnn_*infer.so.8*")
            # Fallback (if no v8 found): any version, but will be validated
            if(NOT _candidates_base)
                file(GLOB _candidates_base "${_d}/libcudnn.so*" "${_d}/libcudnn.so.*")
            endif()
            if(NOT _candidates_infer)
                file(GLOB _candidates_infer "${_d}/libcudnn_*infer.so*" "${_d}/libcudnn_*infer.so.*")
            endif()
            set(_candidates_all ${_candidates_base} ${_candidates_infer})
            if(_candidates_all)
                # Keep only real libs
                set(_filtered)
                foreach(_c ${_candidates_all})
                    _is_real_lib("${_c}" _ok)
                    if(_ok)
                        list(APPEND _filtered "${_c}")
                    endif()
                endforeach()
                if(_filtered)
                    list(SORT _filtered ORDER DESCENDING)
                    list(GET _filtered 0 CUDNN_LIBRARY)
                    break()
                endif()
            endif()
        endif()
    endforeach()
endif()

# Try to find headers (optional; we don't include cudnn.h directly)
if(NOT CUDNN_INCLUDE_DIR)
    if(CUDAToolkit_ROOT)
        foreach(_inc ${CUDAToolkit_ROOT}/include ${CUDAToolkit_ROOT}/targets/x86_64-linux-gnu/include)
            if(EXISTS "${_inc}/cudnn.h" OR EXISTS "${_inc}/cudnn_version.h")
                set(CUDNN_INCLUDE_DIR ${_inc})
                break()
            endif()
        endforeach()
    endif()
endif()

# Build list of cuDNN libs to link (real libs only)
unset(TRT_CUDNN_LIBS)
if(CUDNN_LIBRARY)
    # If a base libcudnn*.so is found, link against it
    _is_real_lib("${CUDNN_LIBRARY}" _ok)
    if(_ok)
        list(APPEND TRT_CUDNN_LIBS ${CUDNN_LIBRARY})
    endif()
else()
    # Otherwise collect component infer libs from candidate dirs
    foreach(_d ${_CUDNN_CANDIDATE_DIRS})
        if(EXISTS "${_d}")
            file(GLOB _infer_libs "${_d}/libcudnn_*infer.so*" "${_d}/libcudnn_*infer.so.*")
            if(_infer_libs)
                foreach(_c ${_infer_libs})
                    _is_real_lib("${_c}" _ok)
                    if(_ok)
                        list(APPEND TRT_CUDNN_LIBS "${_c}")
                    endif()
                endforeach()
            endif()
        endif()
    endforeach()
    if(TRT_CUDNN_LIBS)
        list(REMOVE_DUPLICATES TRT_CUDNN_LIBS)
        list(SORT TRT_CUDNN_LIBS ORDER DESCENDING)
    endif()
endif()

if(TRT_CUDNN_LIBS)
    message(STATUS "cuDNN libraries to link: ${TRT_CUDNN_LIBS}")
    if(CUDNN_INCLUDE_DIR)
        message(STATUS "Found CUDNN include: ${CUDNN_INCLUDE_DIR}")
        include_directories(${CUDNN_INCLUDE_DIR})
    endif()
        # Validate cuDNN major version according to detected TensorRT major
        set(_HAS_V8 FALSE)
        set(_HAS_V9 FALSE)
        foreach(_lib ${TRT_CUDNN_LIBS})
            get_filename_component(_lib_name ${_lib} NAME)
            if(_lib_name MATCHES ".*\\.so\\.8(\\.|$)")
                set(_HAS_V8 TRUE)
            elseif(_lib_name MATCHES ".*\\.so\\.9(\\.|$)")
                set(_HAS_V9 TRUE)
            endif()
        endforeach()
        if(TRT_MAJOR)
            if(TRT_MAJOR EQUAL 8)
                if(NOT _HAS_V8)
                    message(FATAL_ERROR "TensorRT ${TRT_MAJOR}.x detected, but cuDNN v8 not found. Please provide libcudnn*.so.8 (e.g., via Kaggle dataset) or set -DTRT_CUDNN_LIBS to v8 libs.")
                endif()
            elseif(TRT_MAJOR GREATER_EQUAL 9)
                if(NOT _HAS_V9)
                    message(FATAL_ERROR "TensorRT ${TRT_MAJOR}.x detected, but cuDNN v9 not found. Please provide libcudnn*.so.9 or set -DTRT_CUDNN_LIBS to v9 libs.")
                endif()
            endif()
        endif()
else()
    message(WARNING "cuDNN not found. Proceeding without explicit cuDNN linkage. Some TensorRT plugins/layers may fail at runtime.\nChecked in: ${_CUDNN_CANDIDATE_DIRS}.\nTip: set CUDNN_LIBRARY to libcudnn.so.<ver> and/or TRT_CUDNN_LIBS to libcudnn_*infer.so paths (v9 for TensorRT>=9).")
endif()

# TensorRT paths (prefer env-provided, with Kaggle/targets layout support)
# Priority:
#   1) Explicit env: TENSORRT_INCLUDE / TENSORRT_LIBDIR (exported by scripts/setup_kaggle_env.sh)
#   2) Root: TENSORRT_HOME or TENSORRT_ROOT with typical layouts
#   3) NO fallback to generic system dirs unless the header/library actually exist there

set(TENSORRT_INCLUDE_DIRS "" CACHE PATH "TensorRT include directory")
set(TENSORRT_LIBRARY_DIRS "" CACHE PATH "TensorRT library directory")

# 1) Env variables provided by Kaggle/Colab setup scripts
if(DEFINED ENV{TENSORRT_INCLUDE} AND EXISTS "$ENV{TENSORRT_INCLUDE}/NvInfer.h")
    set(TENSORRT_INCLUDE_DIRS "$ENV{TENSORRT_INCLUDE}")
endif()
if(DEFINED ENV{TENSORRT_LIBDIR} AND EXISTS "$ENV{TENSORRT_LIBDIR}/libnvinfer.so")
    set(TENSORRT_LIBRARY_DIRS "$ENV{TENSORRT_LIBDIR}")
endif()

# 2) Derive from TENSORRT_HOME/TENSORRT_ROOT (CMake vars or env) if still not set
if((NOT TENSORRT_INCLUDE_DIRS OR NOT TENSORRT_LIBRARY_DIRS))
    set(_TRT_ROOTS)
    # Prefer CMake cache/command-line variables first
    if(DEFINED TENSORRT_HOME)
        list(APPEND _TRT_ROOTS ${TENSORRT_HOME})
    endif()
    if(DEFINED TENSORRT_ROOT)
        list(APPEND _TRT_ROOTS ${TENSORRT_ROOT})
    endif()
    # Then environment variables
    if(DEFINED ENV{TENSORRT_HOME})
        list(APPEND _TRT_ROOTS $ENV{TENSORRT_HOME})
    endif()
    if(DEFINED ENV{TENSORRT_ROOT})
        list(APPEND _TRT_ROOTS $ENV{TENSORRT_ROOT})
    endif()

    foreach(_ROOT ${_TRT_ROOTS})
        if(TENSORRT_INCLUDE_DIRS AND TENSORRT_LIBRARY_DIRS)
            break()
        endif()
        if(NOT EXISTS "${_ROOT}")
            continue()
        endif()
        # Common layouts under each root
        # Prefer targets/.../include to match the selected libs under targets/.../lib
        set(_TRY_INC
            ${_ROOT}/targets/x86_64-linux-gnu/include
            ${_ROOT}/include
        )
        set(_TRY_LIB
            ${_ROOT}/lib
            ${_ROOT}/targets/x86_64-linux-gnu/lib
        )
        if(NOT TENSORRT_INCLUDE_DIRS)
            foreach(p ${_TRY_INC})
                if(EXISTS "${p}/NvInfer.h")
                    set(TENSORRT_INCLUDE_DIRS ${p})
                    break()
                endif()
            endforeach()
        endif()
        if(NOT TENSORRT_LIBRARY_DIRS)
            foreach(p ${_TRY_LIB})
                if(EXISTS "${p}/libnvinfer.so")
                    set(TENSORRT_LIBRARY_DIRS ${p})
                    break()
                endif()
            endforeach()
        endif()
    endforeach()
endif()

# 3) As a last resort, check system paths but only if actual files exist
if(NOT TENSORRT_INCLUDE_DIRS)
    foreach(p /usr/include/x86_64-linux-gnu /usr/include)
        if(EXISTS "${p}/NvInfer.h")
            set(TENSORRT_INCLUDE_DIRS ${p})
            break()
        endif()
    endforeach()
endif()
if(NOT TENSORRT_LIBRARY_DIRS)
    foreach(p /usr/lib/x86_64-linux-gnu /usr/lib)
        if(EXISTS "${p}/libnvinfer.so")
            set(TENSORRT_LIBRARY_DIRS ${p})
            break()
        endif()
    endforeach()
endif()

if(NOT TENSORRT_INCLUDE_DIRS OR NOT TENSORRT_LIBRARY_DIRS)
    message(FATAL_ERROR "TensorRT not found. Set env via 'source /kaggle/working/tensorrt_env.sh' (Kaggle) or export TENSORRT_HOME/TENSORRT_ROOT.\nInclude='${TENSORRT_INCLUDE_DIRS}' Lib='${TENSORRT_LIBRARY_DIRS}'\nExpected files: NvInfer.h under include, libnvinfer.so under lib.")
else()
    message(STATUS "Found TensorRT include: ${TENSORRT_INCLUDE_DIRS}")
    message(STATUS "Found TensorRT libdir:  ${TENSORRT_LIBRARY_DIRS}")
endif()
include_directories(${TENSORRT_INCLUDE_DIRS})

# Sanity: if libs are from targets/.../lib but include points to top-level include, align include to the sibling targets include
get_filename_component(_TRT_LIB_BASENAME ${TENSORRT_LIBRARY_DIRS} NAME)
if(_TRT_LIB_BASENAME STREQUAL "lib")
    get_filename_component(_TRT_LIB_PARENT ${TENSORRT_LIBRARY_DIRS} DIRECTORY)
    get_filename_component(_TRT_LIB_PARENT_NAME ${_TRT_LIB_PARENT} NAME)
    if(_TRT_LIB_PARENT_NAME STREQUAL "x86_64-linux-gnu")
        # We are inside .../targets/x86_64-linux-gnu/lib
        get_filename_component(_TRT_TARGETS_DIR ${_TRT_LIB_PARENT} DIRECTORY) # .../targets
        set(_TRT_TARGETS_INCLUDE "${_TRT_TARGETS_DIR}/include")
        if(EXISTS "${_TRT_TARGETS_INCLUDE}/NvInfer.h" AND NOT TENSORRT_INCLUDE_DIRS STREQUAL _TRT_TARGETS_INCLUDE)
            if(NOT TENSORRT_INCLUDE_DIRS MATCHES "/targets/.*/include")
                message(WARNING "Adjusting TensorRT include to match libraries: ${_TRT_TARGETS_INCLUDE}")
                set(TENSORRT_INCLUDE_DIRS "${_TRT_TARGETS_INCLUDE}" CACHE PATH "TensorRT include directory" FORCE)
            endif()
        endif()
    endif()
endif()

# OpenCV
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    message(STATUS "Found OpenCV: ${OpenCV_LIBS}")
    include_directories(${OpenCV_INCLUDE_DIRS})
endif()

# --- Find Real TensorRT Library Files ---
# This block handles the Kaggle environment where symlinks like libnvinfer.so are broken (0-byte files).
# It finds the real, versioned .so files and stores them in global variables for all subprojects to use.
if(TENSORRT_LIBRARY_DIRS)
    # Helper: pick newest matching versioned .so if symlinks are broken
    macro(_pick_trt_lib _var _pattern _fallback_name)
        unset(${_var})
        file(GLOB _cand "${TENSORRT_LIBRARY_DIRS}/${_pattern}")
        if(_cand)
            list(SORT _cand ORDER DESCENDING)
            list(GET _cand 0 ${_var})
        endif()
        if(NOT ${_var})
            find_library(${_var} ${_fallback_name} HINTS ${TENSORRT_LIBRARY_DIRS})
        endif()
    endmacro()

    # --- nvinfer ---
    _pick_trt_lib(TRT_NVINFER_LIB "libnvinfer.so.*" nvinfer)

    # --- nvonnxparser ---
    _pick_trt_lib(TRT_NVONNXPARSER_LIB "libnvonnxparser.so.*" nvonnxparser)

    # --- nvinfer_plugin (optional) ---
    _pick_trt_lib(TRT_NVINFER_PLUGIN_LIB "libnvinfer_plugin.so.*" nvinfer_plugin)

    # --- nvinfer_builder_resource (optional) ---
    _pick_trt_lib(TRT_NVINFER_BUILDER_RES_LIB "libnvinfer_builder_resource.so.*" nvinfer_builder_resource)

    if(NOT TRT_NVINFER_LIB)
        message(FATAL_ERROR "Could not find a valid nvinfer library in ${TENSORRT_LIBRARY_DIRS}")
    else()
        message(STATUS "Using nvinfer library: ${TRT_NVINFER_LIB}")
        # Derive TensorRT major version from filename (libnvinfer.so.<MAJOR>...)
        get_filename_component(_TRT_NVINFER_NAME ${TRT_NVINFER_LIB} NAME)
        set(TRT_MAJOR "")
        if(_TRT_NVINFER_NAME MATCHES "libnvinfer\\.so\\.([0-9]+).*")
            set(TRT_MAJOR "${CMAKE_MATCH_1}")
            message(STATUS "Detected TensorRT major version: ${TRT_MAJOR}")
        endif()
    endif()
endif()

# --- Stage runtime libraries into bin (robust against broken symlinks) ---
# Copy only real, versioned .so files, then create clean .so.MAJOR and .so duplicates.
if(TENSORRT_LIBRARY_DIRS)
    # Collect TensorRT real libs (versioned files we already resolved above)
    set(_RUNTIME_LIBS)
    foreach(_trt_lib ${TRT_NVINFER_LIB} ${TRT_NVONNXPARSER_LIB} ${TRT_NVINFER_PLUGIN_LIB} ${TRT_NVINFER_BUILDER_RES_LIB})
        if(_trt_lib)
            # Resolve to real path in case of symlink; skip if resolve fails
            get_filename_component(_real ${_trt_lib} REALPATH)
            if(EXISTS "${_real}")
                list(APPEND _RUNTIME_LIBS ${_real})
            endif()
        endif()
    endforeach()

    # Add cuDNN libs: resolve each to its real path
    if(TRT_CUDNN_LIBS)
        foreach(_lib ${TRT_CUDNN_LIBS})
            if(EXISTS "${_lib}")
                get_filename_component(_real ${_lib} REALPATH)
                if(EXISTS "${_real}")
                    list(APPEND _RUNTIME_LIBS ${_real})
                endif()
            endif()
        endforeach()
    endif()
    if(_RUNTIME_LIBS)
        list(REMOVE_DUPLICATES _RUNTIME_LIBS)
    endif()

    # Helper function: stage a single lib into bin as three files:
    #  - full version filename (as-is), e.g., libnvinfer.so.10.4.0
    #  - major soname, e.g., libnvinfer.so.10
    #  - unversioned, e.g., libnvinfer.so
    function(_stage_one_lib LIBPATH)
        if(NOT EXISTS "${LIBPATH}")
            return()
        endif()
        get_filename_component(_name ${LIBPATH} NAME)            # e.g., libxxx.so.10.4.0
        # Extract root (libxxx), base (libxxx.so), and major (10)
        string(REGEX REPLACE "^(lib[^.]+)\\..*$" "\\1" _root "${_name}")
        set(_base "${_root}.so")
        string(REGEX REPLACE ".*\\.so\\.([0-9]+).*$" "\\1" _maj  "${_name}")
        # Fallback to detected TRT_MAJOR for core TRT libs if major not parsed
        if(NOT _maj MATCHES "^[0-9]+$" AND TRT_MAJOR)
            if(_root STREQUAL "libnvinfer" OR _root STREQUAL "libnvinfer_plugin" OR _root STREQUAL "libnvonnxparser" OR _root STREQUAL "libnvinfer_builder_resource")
                set(_maj "${TRT_MAJOR}")
            endif()
        endif()
        if(NOT _base)
            return()
        endif()
        set(_dst_full  "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${_name}")
        set(_dst_major "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${_base}")
        if(_maj MATCHES "^[0-9]+$")
            set(_dst_soname "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${_base}.${_maj}")
        else()
            set(_dst_soname "")
        endif()
        # Create bin dir
        file(MAKE_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
        # Copy full version file
        add_custom_command(OUTPUT "${_dst_full}"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${LIBPATH}" "${_dst_full}"
            DEPENDS "${LIBPATH}"
            VERBATIM)
        # Duplicate the full version file into .so.MAJOR and .so to avoid symlink issues
        if(_dst_soname)
            add_custom_command(OUTPUT "${_dst_soname}"
                COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_dst_full}" "${_dst_soname}"
                DEPENDS "${_dst_full}"
                VERBATIM)
        endif()
        add_custom_command(OUTPUT "${_dst_major}"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_dst_full}" "${_dst_major}"
            DEPENDS "${_dst_full}"
            VERBATIM)
        # Expose outputs to caller via PARENT_SCOPE to aggregate
        set(STAGED_OUTPUTS ${STAGED_OUTPUTS} "${_dst_full}" PARENT_SCOPE)
        if(_dst_soname)
            set(STAGED_OUTPUTS ${STAGED_OUTPUTS} "${_dst_soname}" PARENT_SCOPE)
        endif()
        set(STAGED_OUTPUTS ${STAGED_OUTPUTS} "${_dst_major}" PARENT_SCOPE)
    endfunction()

    # Generate a phony target that materializes all staged libs
    set(STAGED_OUTPUTS)
    foreach(_L ${_RUNTIME_LIBS})
        _stage_one_lib("${_L}")
    endforeach()
    if(STAGED_OUTPUTS)
        add_custom_target(stage_runtime_libs ALL
            DEPENDS ${STAGED_OUTPUTS}
            COMMENT "Staging TensorRT/cuDNN runtime libraries into bin (no broken symlinks)"
        )
    # Generate a helper env.sh in bin to export LD_LIBRARY_PATH to include bin
    file(MAKE_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
    file(WRITE "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/env.sh"
"#!/usr/bin/env bash\n"
"set -e\n"
"DIR=\"${CMAKE_RUNTIME_OUTPUT_DIRECTORY}\"\n"
"export LD_LIBRARY_PATH=\"$DIR:\\${LD_LIBRARY_PATH}\"\n"
"echo \"LD_LIBRARY_PATH updated to include $DIR\"\n"
    )
    # Note: env.sh is intended to be sourced (source env.sh); execute bit is not required
    else()
    message(WARNING "No runtime libraries collected to stage into bin.")
    endif()
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
