# Library target for shared TensorRT utilities (now physically located under libs/trt_utils)
include(GNUInstallDirs)

# Build options
option(TRT_UTILS_BUILD_SHARED "Build trt_utils as a shared library" OFF)
set(TRT_UTILS_LIB_TYPE STATIC)
if(TRT_UTILS_BUILD_SHARED)
    set(TRT_UTILS_LIB_TYPE SHARED)
endif()

add_library(trt_utils ${TRT_UTILS_LIB_TYPE}
    ${CMAKE_CURRENT_SOURCE_DIR}/src/trt_common.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/trt_builder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/trt_runtime.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/trt_preprocess.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/int8_calibrator.cpp
)

set_target_properties(trt_utils PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Public headers exported from libs/trt_utils/include
target_include_directories(trt_utils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(trt_utils PUBLIC CUDA::cudart ${OpenCV_LIBS})

# Prefer imported TensorRT targets if available, else fall back to library paths provided by parent CMake
if(TARGET TensorRT::nvinfer AND TARGET TensorRT::nvonnxparser)
    target_link_libraries(trt_utils PUBLIC TensorRT::nvinfer TensorRT::nvonnxparser)
else()
    if(DEFINED TENSORRT_LIBRARY_DIRS)
        target_link_libraries(trt_utils PUBLIC
            ${TENSORRT_LIBRARY_DIRS}/libnvinfer.so
            ${TENSORRT_LIBRARY_DIRS}/libnvonnxparser.so)
    else()
        message(WARNING "TensorRT targets not found and TENSORRT_LIBRARY_DIRS not defined. Linking may fail.")
    endif()
endif()

# Export and install package for cross-repo reuse
include(CMakePackageConfigHelpers)
set(TRT_UTILS_VERSION "1.0.0")
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/trt_utilsConfigVersion.cmake
    VERSION ${TRT_UTILS_VERSION}
    COMPATIBILITY SameMajorVersion)

install(TARGETS trt_utils
    EXPORT trt_utilsTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/trt_utils/trt_common.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/trt_utils/trt_builder.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/trt_utils/trt_runtime.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/trt_utils/trt_preprocess.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/trt_utils/int8_calibrator.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/trt_utils)

configure_package_config_file(
    ${PROJECT_SOURCE_DIR}/cmake/trt_utilsConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/trt_utilsConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/trt_utils)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/trt_utilsConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/trt_utilsConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/trt_utils)
install(EXPORT trt_utilsTargets
    NAMESPACE trt_utils::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/trt_utils)
