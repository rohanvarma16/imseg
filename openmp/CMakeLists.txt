cmake_minimum_required(VERSION 2.8)
project( ImgSeg )


FIND_PACKAGE( OpenMP REQUIRED)
if(OPENMP_FOUND)
message("OPENMP FOUND")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

SET("OpenCV_DIR" /afs/cs/academic/class/15418-s17/public/sw/opencv/build)
find_package( OpenCV REQUIRED )
add_executable( main src/main.cpp src/segmentation.cpp src/pdensity.cpp )
target_link_libraries( main ${OpenCV_LIBS} )
