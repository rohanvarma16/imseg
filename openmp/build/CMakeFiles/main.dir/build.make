# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The program to use to edit the cache.
CMAKE_EDIT_COMMAND = /usr/bin/ccmake

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /afs/andrew.cmu.edu/usr18/rohanv/workspace/18645/imseg/openmp

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /afs/andrew.cmu.edu/usr18/rohanv/workspace/18645/imseg/openmp/build

# Include any dependencies generated for this target.
include CMakeFiles/main.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/main.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/main.dir/flags.make

CMakeFiles/main.dir/src/main.cpp.o: CMakeFiles/main.dir/flags.make
CMakeFiles/main.dir/src/main.cpp.o: ../src/main.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /afs/andrew.cmu.edu/usr18/rohanv/workspace/18645/imseg/openmp/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/main.dir/src/main.cpp.o"
	/usr/lib64/ccache/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/main.dir/src/main.cpp.o -c /afs/andrew.cmu.edu/usr18/rohanv/workspace/18645/imseg/openmp/src/main.cpp

CMakeFiles/main.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/main.dir/src/main.cpp.i"
	/usr/lib64/ccache/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /afs/andrew.cmu.edu/usr18/rohanv/workspace/18645/imseg/openmp/src/main.cpp > CMakeFiles/main.dir/src/main.cpp.i

CMakeFiles/main.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/main.dir/src/main.cpp.s"
	/usr/lib64/ccache/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /afs/andrew.cmu.edu/usr18/rohanv/workspace/18645/imseg/openmp/src/main.cpp -o CMakeFiles/main.dir/src/main.cpp.s

CMakeFiles/main.dir/src/main.cpp.o.requires:
.PHONY : CMakeFiles/main.dir/src/main.cpp.o.requires

CMakeFiles/main.dir/src/main.cpp.o.provides: CMakeFiles/main.dir/src/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/main.dir/build.make CMakeFiles/main.dir/src/main.cpp.o.provides.build
.PHONY : CMakeFiles/main.dir/src/main.cpp.o.provides

CMakeFiles/main.dir/src/main.cpp.o.provides.build: CMakeFiles/main.dir/src/main.cpp.o

CMakeFiles/main.dir/src/segmentation.cpp.o: CMakeFiles/main.dir/flags.make
CMakeFiles/main.dir/src/segmentation.cpp.o: ../src/segmentation.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /afs/andrew.cmu.edu/usr18/rohanv/workspace/18645/imseg/openmp/build/CMakeFiles $(CMAKE_PROGRESS_2)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/main.dir/src/segmentation.cpp.o"
	/usr/lib64/ccache/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/main.dir/src/segmentation.cpp.o -c /afs/andrew.cmu.edu/usr18/rohanv/workspace/18645/imseg/openmp/src/segmentation.cpp

CMakeFiles/main.dir/src/segmentation.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/main.dir/src/segmentation.cpp.i"
	/usr/lib64/ccache/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /afs/andrew.cmu.edu/usr18/rohanv/workspace/18645/imseg/openmp/src/segmentation.cpp > CMakeFiles/main.dir/src/segmentation.cpp.i

CMakeFiles/main.dir/src/segmentation.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/main.dir/src/segmentation.cpp.s"
	/usr/lib64/ccache/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /afs/andrew.cmu.edu/usr18/rohanv/workspace/18645/imseg/openmp/src/segmentation.cpp -o CMakeFiles/main.dir/src/segmentation.cpp.s

CMakeFiles/main.dir/src/segmentation.cpp.o.requires:
.PHONY : CMakeFiles/main.dir/src/segmentation.cpp.o.requires

CMakeFiles/main.dir/src/segmentation.cpp.o.provides: CMakeFiles/main.dir/src/segmentation.cpp.o.requires
	$(MAKE) -f CMakeFiles/main.dir/build.make CMakeFiles/main.dir/src/segmentation.cpp.o.provides.build
.PHONY : CMakeFiles/main.dir/src/segmentation.cpp.o.provides

CMakeFiles/main.dir/src/segmentation.cpp.o.provides.build: CMakeFiles/main.dir/src/segmentation.cpp.o

CMakeFiles/main.dir/src/pdensity.cpp.o: CMakeFiles/main.dir/flags.make
CMakeFiles/main.dir/src/pdensity.cpp.o: ../src/pdensity.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /afs/andrew.cmu.edu/usr18/rohanv/workspace/18645/imseg/openmp/build/CMakeFiles $(CMAKE_PROGRESS_3)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/main.dir/src/pdensity.cpp.o"
	/usr/lib64/ccache/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/main.dir/src/pdensity.cpp.o -c /afs/andrew.cmu.edu/usr18/rohanv/workspace/18645/imseg/openmp/src/pdensity.cpp

CMakeFiles/main.dir/src/pdensity.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/main.dir/src/pdensity.cpp.i"
	/usr/lib64/ccache/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /afs/andrew.cmu.edu/usr18/rohanv/workspace/18645/imseg/openmp/src/pdensity.cpp > CMakeFiles/main.dir/src/pdensity.cpp.i

CMakeFiles/main.dir/src/pdensity.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/main.dir/src/pdensity.cpp.s"
	/usr/lib64/ccache/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /afs/andrew.cmu.edu/usr18/rohanv/workspace/18645/imseg/openmp/src/pdensity.cpp -o CMakeFiles/main.dir/src/pdensity.cpp.s

CMakeFiles/main.dir/src/pdensity.cpp.o.requires:
.PHONY : CMakeFiles/main.dir/src/pdensity.cpp.o.requires

CMakeFiles/main.dir/src/pdensity.cpp.o.provides: CMakeFiles/main.dir/src/pdensity.cpp.o.requires
	$(MAKE) -f CMakeFiles/main.dir/build.make CMakeFiles/main.dir/src/pdensity.cpp.o.provides.build
.PHONY : CMakeFiles/main.dir/src/pdensity.cpp.o.provides

CMakeFiles/main.dir/src/pdensity.cpp.o.provides.build: CMakeFiles/main.dir/src/pdensity.cpp.o

# Object files for target main
main_OBJECTS = \
"CMakeFiles/main.dir/src/main.cpp.o" \
"CMakeFiles/main.dir/src/segmentation.cpp.o" \
"CMakeFiles/main.dir/src/pdensity.cpp.o"

# External object files for target main
main_EXTERNAL_OBJECTS =

main: CMakeFiles/main.dir/src/main.cpp.o
main: CMakeFiles/main.dir/src/segmentation.cpp.o
main: CMakeFiles/main.dir/src/pdensity.cpp.o
main: CMakeFiles/main.dir/build.make
main: /afs/cs/academic/class/15418-s17/public/sw/opencv/build/lib/libopencv_calib3d.so.3.2.0
main: /afs/cs/academic/class/15418-s17/public/sw/opencv/build/lib/libopencv_core.so.3.2.0
main: /afs/cs/academic/class/15418-s17/public/sw/opencv/build/lib/libopencv_cudaarithm.so.3.2.0
main: /afs/cs/academic/class/15418-s17/public/sw/opencv/build/lib/libopencv_cudabgsegm.so.3.2.0
main: /afs/cs/academic/class/15418-s17/public/sw/opencv/build/lib/libopencv_cudacodec.so.3.2.0
main: /afs/cs/academic/class/15418-s17/public/sw/opencv/build/lib/libopencv_cudafeatures2d.so.3.2.0
main: /afs/cs/academic/class/15418-s17/public/sw/opencv/build/lib/libopencv_cudafilters.so.3.2.0
main: /afs/cs/academic/class/15418-s17/public/sw/opencv/build/lib/libopencv_cudaimgproc.so.3.2.0
main: /afs/cs/academic/class/15418-s17/public/sw/opencv/build/lib/libopencv_cudalegacy.so.3.2.0
main: /afs/cs/academic/class/15418-s17/public/sw/opencv/build/lib/libopencv_cudaobjdetect.so.3.2.0
main: /afs/cs/academic/class/15418-s17/public/sw/opencv/build/lib/libopencv_cudaoptflow.so.3.2.0
main: /afs/cs/academic/class/15418-s17/public/sw/opencv/build/lib/libopencv_cudastereo.so.3.2.0
main: /afs/cs/academic/class/15418-s17/public/sw/opencv/build/lib/libopencv_cudawarping.so.3.2.0
main: /afs/cs/academic/class/15418-s17/public/sw/opencv/build/lib/libopencv_cudev.so.3.2.0
main: /afs/cs/academic/class/15418-s17/public/sw/opencv/build/lib/libopencv_features2d.so.3.2.0
main: /afs/cs/academic/class/15418-s17/public/sw/opencv/build/lib/libopencv_flann.so.3.2.0
main: /afs/cs/academic/class/15418-s17/public/sw/opencv/build/lib/libopencv_highgui.so.3.2.0
main: /afs/cs/academic/class/15418-s17/public/sw/opencv/build/lib/libopencv_imgcodecs.so.3.2.0
main: /afs/cs/academic/class/15418-s17/public/sw/opencv/build/lib/libopencv_imgproc.so.3.2.0
main: /afs/cs/academic/class/15418-s17/public/sw/opencv/build/lib/libopencv_ml.so.3.2.0
main: /afs/cs/academic/class/15418-s17/public/sw/opencv/build/lib/libopencv_objdetect.so.3.2.0
main: /afs/cs/academic/class/15418-s17/public/sw/opencv/build/lib/libopencv_photo.so.3.2.0
main: /afs/cs/academic/class/15418-s17/public/sw/opencv/build/lib/libopencv_shape.so.3.2.0
main: /afs/cs/academic/class/15418-s17/public/sw/opencv/build/lib/libopencv_stitching.so.3.2.0
main: /afs/cs/academic/class/15418-s17/public/sw/opencv/build/lib/libopencv_superres.so.3.2.0
main: /afs/cs/academic/class/15418-s17/public/sw/opencv/build/lib/libopencv_video.so.3.2.0
main: /afs/cs/academic/class/15418-s17/public/sw/opencv/build/lib/libopencv_videoio.so.3.2.0
main: /afs/cs/academic/class/15418-s17/public/sw/opencv/build/lib/libopencv_videostab.so.3.2.0
main: /afs/cs/academic/class/15418-s17/public/sw/opencv/build/lib/libopencv_aruco.so.3.2.0
main: /afs/cs/academic/class/15418-s17/public/sw/opencv/build/lib/libopencv_bgsegm.so.3.2.0
main: /afs/cs/academic/class/15418-s17/public/sw/opencv/build/lib/libopencv_bioinspired.so.3.2.0
main: /afs/cs/academic/class/15418-s17/public/sw/opencv/build/lib/libopencv_ccalib.so.3.2.0
main: /afs/cs/academic/class/15418-s17/public/sw/opencv/build/lib/libopencv_datasets.so.3.2.0
main: /afs/cs/academic/class/15418-s17/public/sw/opencv/build/lib/libopencv_dnn.so.3.2.0
main: /afs/cs/academic/class/15418-s17/public/sw/opencv/build/lib/libopencv_dpm.so.3.2.0
main: /afs/cs/academic/class/15418-s17/public/sw/opencv/build/lib/libopencv_face.so.3.2.0
main: /afs/cs/academic/class/15418-s17/public/sw/opencv/build/lib/libopencv_freetype.so.3.2.0
main: /afs/cs/academic/class/15418-s17/public/sw/opencv/build/lib/libopencv_fuzzy.so.3.2.0
main: /afs/cs/academic/class/15418-s17/public/sw/opencv/build/lib/libopencv_line_descriptor.so.3.2.0
main: /afs/cs/academic/class/15418-s17/public/sw/opencv/build/lib/libopencv_optflow.so.3.2.0
main: /afs/cs/academic/class/15418-s17/public/sw/opencv/build/lib/libopencv_phase_unwrapping.so.3.2.0
main: /afs/cs/academic/class/15418-s17/public/sw/opencv/build/lib/libopencv_plot.so.3.2.0
main: /afs/cs/academic/class/15418-s17/public/sw/opencv/build/lib/libopencv_reg.so.3.2.0
main: /afs/cs/academic/class/15418-s17/public/sw/opencv/build/lib/libopencv_rgbd.so.3.2.0
main: /afs/cs/academic/class/15418-s17/public/sw/opencv/build/lib/libopencv_saliency.so.3.2.0
main: /afs/cs/academic/class/15418-s17/public/sw/opencv/build/lib/libopencv_stereo.so.3.2.0
main: /afs/cs/academic/class/15418-s17/public/sw/opencv/build/lib/libopencv_structured_light.so.3.2.0
main: /afs/cs/academic/class/15418-s17/public/sw/opencv/build/lib/libopencv_surface_matching.so.3.2.0
main: /afs/cs/academic/class/15418-s17/public/sw/opencv/build/lib/libopencv_text.so.3.2.0
main: /afs/cs/academic/class/15418-s17/public/sw/opencv/build/lib/libopencv_tracking.so.3.2.0
main: /afs/cs/academic/class/15418-s17/public/sw/opencv/build/lib/libopencv_xfeatures2d.so.3.2.0
main: /afs/cs/academic/class/15418-s17/public/sw/opencv/build/lib/libopencv_ximgproc.so.3.2.0
main: /afs/cs/academic/class/15418-s17/public/sw/opencv/build/lib/libopencv_xobjdetect.so.3.2.0
main: /afs/cs/academic/class/15418-s17/public/sw/opencv/build/lib/libopencv_xphoto.so.3.2.0
main: /afs/cs/academic/class/15418-s17/public/sw/opencv/build/lib/libopencv_cudafeatures2d.so.3.2.0
main: /afs/cs/academic/class/15418-s17/public/sw/opencv/build/lib/libopencv_shape.so.3.2.0
main: /afs/cs/academic/class/15418-s17/public/sw/opencv/build/lib/libopencv_cudacodec.so.3.2.0
main: /afs/cs/academic/class/15418-s17/public/sw/opencv/build/lib/libopencv_cudaoptflow.so.3.2.0
main: /afs/cs/academic/class/15418-s17/public/sw/opencv/build/lib/libopencv_cudalegacy.so.3.2.0
main: /afs/cs/academic/class/15418-s17/public/sw/opencv/build/lib/libopencv_cudawarping.so.3.2.0
main: /afs/cs/academic/class/15418-s17/public/sw/opencv/build/lib/libopencv_photo.so.3.2.0
main: /afs/cs/academic/class/15418-s17/public/sw/opencv/build/lib/libopencv_cudaimgproc.so.3.2.0
main: /afs/cs/academic/class/15418-s17/public/sw/opencv/build/lib/libopencv_cudafilters.so.3.2.0
main: /afs/cs/academic/class/15418-s17/public/sw/opencv/build/lib/libopencv_cudaarithm.so.3.2.0
main: /afs/cs/academic/class/15418-s17/public/sw/opencv/build/lib/libopencv_calib3d.so.3.2.0
main: /afs/cs/academic/class/15418-s17/public/sw/opencv/build/lib/libopencv_phase_unwrapping.so.3.2.0
main: /afs/cs/academic/class/15418-s17/public/sw/opencv/build/lib/libopencv_video.so.3.2.0
main: /afs/cs/academic/class/15418-s17/public/sw/opencv/build/lib/libopencv_datasets.so.3.2.0
main: /afs/cs/academic/class/15418-s17/public/sw/opencv/build/lib/libopencv_dnn.so.3.2.0
main: /afs/cs/academic/class/15418-s17/public/sw/opencv/build/lib/libopencv_plot.so.3.2.0
main: /afs/cs/academic/class/15418-s17/public/sw/opencv/build/lib/libopencv_text.so.3.2.0
main: /afs/cs/academic/class/15418-s17/public/sw/opencv/build/lib/libopencv_features2d.so.3.2.0
main: /afs/cs/academic/class/15418-s17/public/sw/opencv/build/lib/libopencv_flann.so.3.2.0
main: /afs/cs/academic/class/15418-s17/public/sw/opencv/build/lib/libopencv_highgui.so.3.2.0
main: /afs/cs/academic/class/15418-s17/public/sw/opencv/build/lib/libopencv_ml.so.3.2.0
main: /afs/cs/academic/class/15418-s17/public/sw/opencv/build/lib/libopencv_videoio.so.3.2.0
main: /afs/cs/academic/class/15418-s17/public/sw/opencv/build/lib/libopencv_imgcodecs.so.3.2.0
main: /afs/cs/academic/class/15418-s17/public/sw/opencv/build/lib/libopencv_objdetect.so.3.2.0
main: /afs/cs/academic/class/15418-s17/public/sw/opencv/build/lib/libopencv_imgproc.so.3.2.0
main: /afs/cs/academic/class/15418-s17/public/sw/opencv/build/lib/libopencv_core.so.3.2.0
main: /afs/cs/academic/class/15418-s17/public/sw/opencv/build/lib/libopencv_cudev.so.3.2.0
main: CMakeFiles/main.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable main"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/main.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/main.dir/build: main
.PHONY : CMakeFiles/main.dir/build

CMakeFiles/main.dir/requires: CMakeFiles/main.dir/src/main.cpp.o.requires
CMakeFiles/main.dir/requires: CMakeFiles/main.dir/src/segmentation.cpp.o.requires
CMakeFiles/main.dir/requires: CMakeFiles/main.dir/src/pdensity.cpp.o.requires
.PHONY : CMakeFiles/main.dir/requires

CMakeFiles/main.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/main.dir/cmake_clean.cmake
.PHONY : CMakeFiles/main.dir/clean

CMakeFiles/main.dir/depend:
	cd /afs/andrew.cmu.edu/usr18/rohanv/workspace/18645/imseg/openmp/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /afs/andrew.cmu.edu/usr18/rohanv/workspace/18645/imseg/openmp /afs/andrew.cmu.edu/usr18/rohanv/workspace/18645/imseg/openmp /afs/andrew.cmu.edu/usr18/rohanv/workspace/18645/imseg/openmp/build /afs/andrew.cmu.edu/usr18/rohanv/workspace/18645/imseg/openmp/build /afs/andrew.cmu.edu/usr18/rohanv/workspace/18645/imseg/openmp/build/CMakeFiles/main.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/main.dir/depend

