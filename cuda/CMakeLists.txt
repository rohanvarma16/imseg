cmake_minimum_required(VERSION 2.8)
project( ImgSeg )

find_package(CUDA REQUIRED)
set(CUDA_SEPARABLE_COMPILATION ON)
IF(CUDA_FOUND)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11" )
INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})
SET(LIBRARIES ${CUDA_LIBRARIES})
cuda_compile(cuda_methods src/cuda_methods.cu)
cuda_add_library(cuda_methods_lib src/cuda_methods.cu)
ENDIF()

SET("OpenCV_DIR" /afs/cs/academic/class/15418-s17/public/sw/opencv/build)
find_package( OpenCV REQUIRED )
add_executable( main src/main.cpp src/segmentation.cpp src/pdensity.cpp )
target_link_libraries( main cuda_methods_lib ${CUDA_LIBRARIES} ${OpenCV_LIBS}  )
