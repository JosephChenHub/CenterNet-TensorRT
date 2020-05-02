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

cmake_minimum_required(VERSION 3.13)
project(trt_infer LANGUAGES CXX C)

set(TRTINFER_ROOT ${PROJECT_SOURCE_DIR})
set(DET_EXE_NAME ctdet_infer)
set(POSE_EXE_NAME pose_infer)

# c++11
set(CMAKE_CXX_STANDARD  11)

# Set debug or release  mode
#set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_BUILD_TYPE "Release")

set(USE_LIBTORCH OFF)
set(USE_OPENCV ON)
set(BUILD_GTEST OFF)
set(USE_TENSORRT ON)


# Enable compiler warnings
if (CMAKE_BUILD_TYPE MATCHES "Debug")
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wall -ggdb -O0 -Wno-deprecated-declarations -Wno-unused-function ")
else(CMAKE_BUILD_TYPE "Release")
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wall -O2 -Wno-deprecated-declarations -Wno-unused-function")
endif()


# Build the libraries with -fPIC
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

#--------------------------------------------------
# Version information
#--------------------------------------------------

#--------------------------------------------------
# Build configurations, global to all projects
#--------------------------------------------------

set(DET_EXE_SOURCES
  ${PROJECT_SOURCE_DIR}/src/ctdet_infer.cpp
  ${PROJECT_SOURCE_DIR}/common/logger.cpp
)
set(POSE_EXE_SOURCES
  ${PROJECT_SOURCE_DIR}/src/pose_infer.cpp
  ${PROJECT_SOURCE_DIR}/common/logger.cpp
)

set(CU_SOURCES
    ${PROJECT_SOURCE_DIR}/src/det_kernels.cu
    #${PROJECT_SOURCE_DIR}/src/topk_gpu.cu
    ${PROJECT_SOURCE_DIR}/src/gpu_sort.cu
    ${PROJECT_SOURCE_DIR}/src/custom.cpp
    #${PROJECT_SOURCE_DIR}/src/warp_affine.cu
)

set(EXE_INCLUDE
    ${PROJECT_SOURCE_DIR}/tensorrt-plugin/DCN
    ${PROJECT_SOURCE_DIR}/common
    ${PROJECT_SOURCE_DIR}/include
)

set(CU_INCLUDE
    ${PROJECT_SOURCE_DIR}/include
)

#if (NOT TARGET protobuf::libprotobuf)
#  FIND_PACKAGE(Protobuf REQUIRED)
#else()
#  set(PROTOBUF_LIB "protobuf::libprotobuf")
#endif()


#
find_package(CUDA 10.0 REQUIRED)

# TensorRT
if (USE_TENSORRT)
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
#find_library(TENSORRT_LIBRARY_MYELIN myelin
#  HINTS  ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
#  PATH_SUFFIXES lib lib64 lib/x64)
set(TENSORRT_LIBRARY ${TENSORRT_LIBRARY_INFER} ${TENSORRT_LIBRARY_INFER_PLUGIN}) # ${TENSORRT_LIBRARY_MYELIN})
MESSAGE(STATUS "Find TensorRT libs at ${TENSORRT_LIBRARY}")
find_package_handle_standard_args(
  TENSORRT DEFAULT_MSG TENSORRT_INCLUDE_DIR TENSORRT_LIBRARY)
if(NOT TENSORRT_FOUND)
  message(ERROR "Cannot find TensorRT library.")
endif()
endif(USE_TENSORRT)

find_library(CUDNN_LIBRARY libcudnn.so HINTS ${CUDNN_ROOT} lib lib64)
message("FOUND:${CUDNN_LIBRARY}")

if(USE_OPENCV)
find_package(OpenCV 3.0 REQUIRED)
if(OpenCV_FOUND)
    message("Found OpenCV:${OpenCV_INCLUDE_DIRS}, libs:${OpenCV_LIBS}")
endif()
endif(USE_OPENCV)


include_directories(${TENSORRT_INCLUDE_DIR} ${CUDA_INCLUDE_DIRS})

# CUDA
list(APPEND CUDA_NVCC_FLAGS "-Xcompiler -fPIC --expt-extended-lambda --expt-relaxed-constexpr -std=c++11 -O2")
CUDA_INCLUDE_DIRECTORIES(${CUDNN_INCLUDE_DIR} ${TENSORRT_INCLUDE_DIR})
CUDA_ADD_LIBRARY(cu_lib SHARED ${CU_SOURCES})
target_link_libraries(cu_lib ${OpenCV_LIBS})
target_include_directories(cu_lib PUBLIC ${CU_INCLUDE})


# Torch
if(USE_LIBTORCH)
    find_package(Torch REQUIRED)
    message("TORCH_CXX_FLAGS:${TORCH_CXX_FLAGS}!")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
    #set(CMAKE_CXX_STANDARD 14)
    message("Torch libs:${TORCH_LIBRARIES}")
    message("Torch flags:${TORCH_CXX_FLAGS}")
    message("CMAKE_CXX_FLAGS:${CMAKE_CXX_FLAGS}")
endif()




# --------------------------------
# GTEST
# --------------------------------
if(BUILD_GTEST)
    find_package(GTest REQUIRED)
    if(GTest_FOUND)
        message("Found GTest!")
    endif()
    set(KERNEL_NAME modules)
    set(GTEST_LISTS
        ${PROJECT_SOURCE_DIR}/src/test.cc
        ${PROJECT_SOURCE_DIR}/src/topk_cpu.cpp
        ${PROJECT_SOURCE_DIR}/src/custom.cpp
        )
endif()

if(BUILD_GTEST)
        message("GTEST_SOURCES:${GTEST_SOURCES}")
        #foreach(GTEST_LISTS ${GTEST_SOURCES})
            message("Found GTest sources:${GTEST_LISTS}")
            add_executable(gtest_${KERNEL_NAME} ${GTEST_LISTS})

            target_include_directories(gtest_${KERNEL_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/include)
		    target_link_libraries(gtest_${KERNEL_NAME} ${OpenCV_LIBS} cu_lib)
		    if(USE_LIBTORCH)
		        target_link_libraries(gtest_${KERNEL_NAME} ${TORCH_LIBRARIES} ${CUDA_LIBRARIES})
	        else(NOT USE_LIBTORCH)
                target_link_libraries(gtest_${KERNEL_NAME} ${CUDA_LIBRARIES})
		    endif(USE_LIBTORCH)

            target_link_libraries(gtest_${KERNEL_NAME} GTest::GTest GTest::Main)
            add_test(AllTestsInFoo gtest_${KERNEL_NAME})
            #endforeach(GTEST_LISTS)
endif()



# --------------------------------
# Converter executable
# --------------------------------
add_executable(${DET_EXE_NAME} ${DET_EXE_SOURCES})
target_include_directories(${DET_EXE_NAME} PUBLIC ${EXE_INCLUDE})
target_link_libraries(${DET_EXE_NAME} ${TENSORRT_LIBRARY}
    ${CUDA_LIBRARIES} ${CUDNN_LIBRARY})
target_link_libraries(${DET_EXE_NAME} cu_lib)
target_link_libraries(${DET_EXE_NAME} ${PROJECT_SOURCE_DIR}/tensorrt-plugin/build/libDCN.so)


if(USE_OPENCV)
    target_include_directories(${DET_EXE_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(${DET_EXE_NAME} ${OpenCV_LIBS})
endif()

add_executable(${POSE_EXE_NAME} ${POSE_EXE_SOURCES})
target_include_directories(${POSE_EXE_NAME} PUBLIC ${EXE_INCLUDE})
target_link_libraries(${POSE_EXE_NAME} ${TENSORRT_LIBRARY}
    ${CUDA_LIBRARIES} ${CUDNN_LIBRARY})
target_link_libraries(${POSE_EXE_NAME} cu_lib)
target_link_libraries(${POSE_EXE_NAME} ${PROJECT_SOURCE_DIR}/tensorrt-plugin/build/libDCN.so)


if(USE_OPENCV)
    target_include_directories(${POSE_EXE_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(${POSE_EXE_NAME} ${OpenCV_LIBS})
endif()
