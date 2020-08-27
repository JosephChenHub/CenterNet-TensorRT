 # Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 #
 # Permission is hereby granted, free of charge, to any person obtaining a
 # copy of this software and associated documentation files (the "Software"),
 # to deal in the Software without restriction, including without limitation
 # the rights to use, copy, modify, merge, publish, distribute, sublicense,
 # and/or sell copies of the Software, and to permit persons to whom the
 # Software is furnished to do so, subject to the following conditions:
 #
 # The above copyright notice and this permission notice shall be included in
 # all copies or substantial portions of the Software.
 #
 # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 # THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 # FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 # DEALINGS IN THE SOFTWARE.

cmake_minimum_required(VERSION 3.10)
project(onnx2trt LANGUAGES CXX C)

set(ONNX2TRT_ROOT ${PROJECT_SOURCE_DIR})
# Set C++11 as standard for the whole project
set(CMAKE_CXX_STANDARD  11)
#set(PYBIND11_CPP_STANDARD -std=c++11) #pybind11

#find_package(pybind11 REQUIRED)

#SET(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_BUILD_TYPE "Release")

# Enable compiler warnings
if (CMAKE_COMPILER_IS_GNUCC)
    if(CMAKE_BUILD_TYPE MATCHES "Debug")
        set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wall -ggdb -O0 -Wno-deprecated-declarations -Wno-unused-function ")
    else(CMAKE_BUILD_TYPE MATCHES "Release")
        set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wall -O2 -Wno-deprecated-declarations -Wno-unused-function")
    endif()
endif()
if (MSVC)
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} /W4")
endif()

# Build the libraries with -fPIC
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(PARSER_LINKER_SCRIPT  ${ONNX2TRT_ROOT}/libnvonnxparser.version)
set(RUNTIME_LINKER_SCRIPT ${ONNX2TRT_ROOT}/libnvonnxparser_runtime.version)
message("parser_linker_script:${PARSER_LINKER_SCRIPT}")
message("runtime_linker_script:${RUNTIME_LINKER_SCRIPT}")
#--------------------------------------------------
# Version information
#--------------------------------------------------
set(ONNX2TRT_MAJOR 7)
set(ONNX2TRT_MINOR 0)
set(ONNX2TRT_PATCH 0)

#--------------------------------------------------
# Build configurations, global to all projects
#--------------------------------------------------
#set(PLUGIN_SOURCES
    #    ResizeNearest.cu
    #    ${PROJECT_SOURCE_DIR}/plugin/ResizeBilinear.cu
#   plugin.cpp
#)

set(IMPORTER_SOURCES
  NvOnnxParser.cpp
  ModelImporter.cpp
  builtin_op_importers.cpp
  onnx2trt_utils.cpp
  ShapedWeights.cpp
  ShapeTensor.cpp
  OnnxAttrs.cpp
)
#set(RUNTIME_SOURCES
#  NvOnnxParserRuntime.cpp
#  PluginFactory.cpp
#  builtin_plugins.cpp
#)

# Do not build ONNXIFI by default.
if(BUILD_ONNXIFI)
  if (NOT CUDA_TOOLKIT_ROOT_DIR)
    set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
  endif()
  message(debug "CUDA_TOOLKIT_ROOT_DIR: ${CUDA_TOOLKIT_ROOT_DIR}")
  find_path(CUDA_INCLUDE_DIR cuda_runtime.h
    HINTS ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES include
  )
  set(ONNXIFI_SOURCES onnx_trt_backend.cpp)
endif()

set(EXECUTABLE_SOURCES
  main.cpp
)

set(API_TESTS_SOURCES
  getSupportedAPITest.cpp
  ModelImporter.cpp
)

set(HEADERS
  NvOnnxParser.h
  #  NvOnnxParserRuntime.h
)

if (NOT TARGET protobuf::libprotobuf)
  FIND_PACKAGE(Protobuf REQUIRED)
else()
  set(PROTOBUF_LIB "protobuf::libprotobuf")
endif()

if(NOT TARGET onnx_proto)
  # Note: This avoids libprotobuf.so complaining about name collisions at runtime
  if(NOT ONNX_NAMESPACE)
    set(ONNX_NAMESPACE "onnx2trt_onnx")
  endif()
  add_definitions("-DONNX_NAMESPACE=${ONNX_NAMESPACE}")
  add_subdirectory(third_party/onnx EXCLUDE_FROM_ALL)
endif()

# TensorRT
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES include)
MESSAGE(STATUS "Found TensorRT headers at ${TENSORRT_INCLUDE_DIR}")
find_library(TENSORRT_LIBRARY_INFER nvinfer
  HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64)
find_library(TENSORRT_LIBRARY_INFER_PLUGIN nvinfer_plugin
  HINTS  ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64)
find_library(TENSORRT_LIBRARY_MYELIN myelin
  HINTS  ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64)
set(TENSORRT_LIBRARY ${TENSORRT_LIBRARY_INFER} ${TENSORRT_LIBRARY_INFER_PLUGIN} ${TENSORRT_LIBRARY_MYELIN})
MESSAGE(STATUS "Find TensorRT libs at ${TENSORRT_LIBRARY}")
find_package_handle_standard_args(
  TENSORRT DEFAULT_MSG TENSORRT_INCLUDE_DIR TENSORRT_LIBRARY)
if(NOT TENSORRT_FOUND)
  message(ERROR "Cannot find TensorRT library.")
endif()

# ------------------------------------------------------------
# Plugin library, compile the plugins beforehanded
# ------------------------------------------------------------
##set(nvonnxparser_plugin "${PROJECT_SOURCE_DIR}/plugin/build/libcustom_v0.so")
#include_directories(${PROJECT_SOURCE_DIR}/plugin)
##if(NOT nvonnxparser_plugin)
#message("compile the plugin!")
#find_package(CUDA 10.0 REQUIRED)
#list(APPEND CUDA_NVCC_FLAGS "-Xcompiler -fPIC --expt-extended-lambda -std=c++11")
#CUDA_INCLUDE_DIRECTORIES(${CUDNN_INCLUDE_DIR} ${TENSORRT_INCLUDE_DIR})
#CUDA_ADD_LIBRARY(nvonnxparser_plugin ${PLUGIN_SOURCES})
##
#target_include_directories(nvonnxparser_plugin PUBLIC ${CUDA_INCLUDE_DIRS} ${ONNX_INCLUDE_DIRS} ${TENSORRT_INCLUDE_DIR} ${CUDNN_INCLUDE_DIR})
#target_link_libraries(nvonnxparser_plugin ${TENSORRT_LIBRARY})
#endif()

set(PLUGIN_NAMES
    Resize_v0
    DCN
)

foreach(PLUGIN_NAME ${PLUGIN_NAMES})
    set(PLUGIN_INCLUDE ${PLUGIN_INCLUDE} ${PROJECT_SOURCE_DIR}/plugin/${PLUGIN_NAME})
    set(nvonnxparser_plugin ${nvonnxparser_plugin} ${PROJECT_SOURCE_DIR}/plugin/build/lib${PLUGIN_NAME}.so)
endforeach(PLUGIN_NAME)

# --------------------------------
# Importer library
# --------------------------------

add_library(nvonnxparser SHARED ${IMPORTER_SOURCES})
target_include_directories(nvonnxparser PUBLIC ${ONNX_INCLUDE_DIRS} ${TENSORRT_INCLUDE_DIR} ${PLUGIN_INCLUDE})
target_link_libraries(nvonnxparser PUBLIC onnx_proto ${nvonnxparser_plugin} ${PROTOBUF_LIBRARY} ${TENSORRT_LIBRARY})
set_target_properties(nvonnxparser PROPERTIES
  VERSION   ${ONNX2TRT_MAJOR}.${ONNX2TRT_MINOR}.${ONNX2TRT_PATCH}
  SOVERSION ${ONNX2TRT_MAJOR}
  LINK_DEPENDS ${PARSER_LINKER_SCRIPT}
  LINK_FLAGS "-Wl,--version-script=${PARSER_LINKER_SCRIPT}"
)
add_library(nvonnxparser_static STATIC ${IMPORTER_SOURCES})
target_include_directories(nvonnxparser_static PUBLIC ${ONNX_INCLUDE_DIRS} ${TENSORRT_INCLUDE_DIR} ${PLUGIN_INCLUDE})
target_link_libraries(nvonnxparser_static PUBLIC onnx_proto ${nvonnxparser_plugin} ${PROTOBUF_LIBRARY} ${TENSORRT_LIBRARY})

# --------------------------------
# Runtime library
# --------------------------------
#add_library(nvonnxparser_runtime SHARED ${RUNTIME_SOURCES})
#target_include_directories(nvonnxparser_runtime PUBLIC ${CUDA_INCLUDE_DIRS} ${TENSORRT_INCLUDE_DIR} ${CUDNN_INCLUDE_DIR})
#target_link_libraries(nvonnxparser_runtime PUBLIC ${nvonnxparser_plugin} ${CUDNN_LIBRARY} ${TENSORRT_LIBRARY})
#set_target_properties(nvonnxparser_runtime PROPERTIES
#  VERSION   ${ONNX2TRT_MAJOR}.${ONNX2TRT_MINOR}.${ONNX2TRT_PATCH}
#  SOVERSION ${ONNX2TRT_MAJOR}
#  LINK_DEPENDS ${RUNTIME_LINKER_SCRIPT}
#  LINK_FLAGS "-Wl,--version-script=${RUNTIME_LINKER_SCRIPT}"
#)
#add_library(nvonnxparser_runtime_static STATIC ${RUNTIME_SOURCES})
#target_include_directories(nvonnxparser_runtime_static PUBLIC ${CUDA_INCLUDE_DIRS} ${TENSORRT_INCLUDE_DIR} ${CUDNN_INCLUDE_DIR})
#target_link_libraries(nvonnxparser_runtime_static PUBLIC ${nvonnxparser_plugin} ${CUDNN_LIBRARY} ${TENSORRT_LIBRARY})

# --------------------------------
# Onnxifi library
# --------------------------------
if(BUILD_ONNXIFI)
  add_library(trt_onnxify SHARED ${ONNXIFI_SOURCES})
  target_include_directories(trt_onnxify PUBLIC ${CUDA_INCLUDE_DIR} ${ONNX_INCLUDE_DIRS} ${TENSORRT_INCLUDE_DIR})
  target_link_libraries(trt_onnxify PUBLIC nvonnxparser_static ${CMAKE_THREAD_LIBS_INIT} ${CMAKE_DL_LIBS})
endif()

## --------------------------------
# Converter executable
# --------------------------------
add_executable(onnx2trt ${EXECUTABLE_SOURCES})
target_include_directories(onnx2trt PUBLIC ${ONNX_INCLUDE_DIRS} ${CUDNN_INCLUDE_DIR})
target_link_libraries(onnx2trt PUBLIC ${PROTOBUF_LIB} nvonnxparser_static ${CMAKE_THREAD_LIBS_INIT} ${CMAKE_DL_LIBS} ${CUDA_LIBRARIES})

include_directories(${TENSORRT_INCLUDE_DIR})


# --------------------------------
# API Tests
# --------------------------------
add_executable(getSupportedAPITest ${API_TESTS_SOURCES})
target_include_directories(getSupportedAPITest PUBLIC ${ONNX_INCLUDE_DIRS} ${CUDNN_INCLUDE_DIR})
target_link_libraries(getSupportedAPITest PUBLIC ${PROTOBUF_LIB} nvonnxparser_static ${CMAKE_THREAD_LIBS_INIT} ${CMAKE_DL_LIBS}) #${CUDA_LIBRARIES}

# --------------------------------
# Installation
# --------------------------------

#find_package (Python3 3.5 COMPONENTS Interpreter Development REQUIRED)
##find_path(PYTHON3_INC_DIR Python.h HINTS ${PYTHON_ROOT} PATH_SUFFIXES python3.7 python3.6 python3.5 python3.4)
#
#pybind11_add_module(MyUpsample SHARED THIN_LTO ResizeBind.cpp)
#target_include_directories(MyUpsample BEFORE PUBLIC ${Python3_INCLUDE_DIRS})
#message("Python3_inc:${Python3_INCLUDE_DIRS}")
#message("PYTHON3_LIBS:${Python3_LIBRARIES}")
#target_link_libraries(MyUpsample PRIVATE nvonnxparser_static pybind11::module ${Python3_LIBRARIES})
#target_include_directories(MyUpsample PUBLIC ${TENSORRT_INCLUDE_DIR})

install(TARGETS onnx2trt
                nvonnxparser    #MyUpsample        #nvonnxparser_runtime
                nvonnxparser_static #nvonnxparser_runtime_static
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
)

install(FILES ${HEADERS}
  DESTINATION include
)

SET(CPACK_GENERATOR "DEB")
SET(CPACK_DEBIAN_PACKAGE_MAINTAINER "Mike Houston") #required
SET(CPACK_PACKAGE_NAME "onnx-trt-dev")
SET(CPACK_PACKAGE_VERSION "0.5.9")
SET(CPACK_PACKAGE_VERSION_MAJOR "0")
SET(CPACK_PACKAGE_VERSION_MINOR "5")
SET(CPACK_PACKAGE_VERSION_PATCH "9")

INCLUDE(CPack)
