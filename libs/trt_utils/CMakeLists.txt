# Library target for shared TensorRT utilities
include(GNUInstallDirs)

# The library type is now fixed to SHARED, controlled by the root CMakeLists.txt
add_library(trt_utils SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/src/trt_common.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/trt_builder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/trt_runtime.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/trt_preprocess.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/int8_calibrator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/trt_vision.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/trt_decode.cpp
)

# For a shared library, ensure it has the necessary RPATH to find its own dependencies at runtime
set_target_properties(trt_utils PROPERTIES
    INSTALL_RPATH "$ORIGIN"
)

set_target_properties(trt_utils PROPERTIES POSITION_INDEPENDENT_CODE ON)


# Public headers exported from libs/trt_utils/include
target_include_directories(trt_utils
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# Link against foundational libraries.
# By specifying PUBLIC, any target that links against trt_utils will automatically
# also link against these libraries in the correct order.
target_link_libraries(trt_utils PUBLIC CUDA::cudart ${OpenCV_LIBS})

# Add TensorRT headers to this target include path
if(DEFINED TENSORRT_INCLUDE_DIRS)
    target_include_directories(trt_utils PUBLIC ${TENSORRT_INCLUDE_DIRS})
endif()

# Link against the real TensorRT library files found by the root CMakeLists.txt
# Using PUBLIC ensures that executables linking trt_utils get these dependencies.
if(TRT_NVINFER_LIB)
    target_link_libraries(trt_utils PUBLIC ${TRT_NVINFER_LIB})
else()
    message(FATAL_ERROR "TRT_NVINFER_LIB not defined by root CMakeLists.txt")
endif()
if(TRT_NVONNXPARSER_LIB)
    target_link_libraries(trt_utils PUBLIC ${TRT_NVONNXPARSER_LIB})
endif()
if(TRT_NVINFER_PLUGIN_LIB)
    target_link_libraries(trt_utils PUBLIC ${TRT_NVINFER_PLUGIN_LIB})
endif()
if(TRT_NVINFER_BUILDER_RES_LIB)
    target_link_libraries(trt_utils PUBLIC ${TRT_NVINFER_BUILDER_RES_LIB})
endif()
if(TRT_CUDNN_LIBS)
    target_link_libraries(trt_utils PUBLIC ${TRT_CUDNN_LIBS})
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
    ${CMAKE_CURRENT_SOURCE_DIR}/include/trt_utils/trt_vision.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/trt_utils/trt_decode.h
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
