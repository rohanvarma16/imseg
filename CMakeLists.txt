cmake_minimum_required(VERSION 2.8)
project( ImgSeg )
SET("OpenCV_DIR" /afs/cs/academic/class/15418-s17/public/sw/opencv/build)
find_package( OpenCV REQUIRED )
add_executable( main src/main.cpp src/segmentation.cpp src/pdensity.cpp )
target_link_libraries( main ${OpenCV_LIBS} )
