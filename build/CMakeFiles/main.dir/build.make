# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.7

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


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
CMAKE_COMMAND = /usr/local/Cellar/cmake/3.7.2/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/3.7.2/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/rohan/Dropbox/Work/workspace/imseg

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/rohan/Dropbox/Work/workspace/imseg/build

# Include any dependencies generated for this target.
include CMakeFiles/main.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/main.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/main.dir/flags.make

CMakeFiles/main.dir/src/main.cpp.o: CMakeFiles/main.dir/flags.make
CMakeFiles/main.dir/src/main.cpp.o: ../src/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/rohan/Dropbox/Work/workspace/imseg/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/main.dir/src/main.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/main.dir/src/main.cpp.o -c /Users/rohan/Dropbox/Work/workspace/imseg/src/main.cpp

CMakeFiles/main.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/main.dir/src/main.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/rohan/Dropbox/Work/workspace/imseg/src/main.cpp > CMakeFiles/main.dir/src/main.cpp.i

CMakeFiles/main.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/main.dir/src/main.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/rohan/Dropbox/Work/workspace/imseg/src/main.cpp -o CMakeFiles/main.dir/src/main.cpp.s

CMakeFiles/main.dir/src/main.cpp.o.requires:

.PHONY : CMakeFiles/main.dir/src/main.cpp.o.requires

CMakeFiles/main.dir/src/main.cpp.o.provides: CMakeFiles/main.dir/src/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/main.dir/build.make CMakeFiles/main.dir/src/main.cpp.o.provides.build
.PHONY : CMakeFiles/main.dir/src/main.cpp.o.provides

CMakeFiles/main.dir/src/main.cpp.o.provides.build: CMakeFiles/main.dir/src/main.cpp.o


CMakeFiles/main.dir/src/segmentation.cpp.o: CMakeFiles/main.dir/flags.make
CMakeFiles/main.dir/src/segmentation.cpp.o: ../src/segmentation.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/rohan/Dropbox/Work/workspace/imseg/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/main.dir/src/segmentation.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/main.dir/src/segmentation.cpp.o -c /Users/rohan/Dropbox/Work/workspace/imseg/src/segmentation.cpp

CMakeFiles/main.dir/src/segmentation.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/main.dir/src/segmentation.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/rohan/Dropbox/Work/workspace/imseg/src/segmentation.cpp > CMakeFiles/main.dir/src/segmentation.cpp.i

CMakeFiles/main.dir/src/segmentation.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/main.dir/src/segmentation.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/rohan/Dropbox/Work/workspace/imseg/src/segmentation.cpp -o CMakeFiles/main.dir/src/segmentation.cpp.s

CMakeFiles/main.dir/src/segmentation.cpp.o.requires:

.PHONY : CMakeFiles/main.dir/src/segmentation.cpp.o.requires

CMakeFiles/main.dir/src/segmentation.cpp.o.provides: CMakeFiles/main.dir/src/segmentation.cpp.o.requires
	$(MAKE) -f CMakeFiles/main.dir/build.make CMakeFiles/main.dir/src/segmentation.cpp.o.provides.build
.PHONY : CMakeFiles/main.dir/src/segmentation.cpp.o.provides

CMakeFiles/main.dir/src/segmentation.cpp.o.provides.build: CMakeFiles/main.dir/src/segmentation.cpp.o


CMakeFiles/main.dir/src/pdensity.cpp.o: CMakeFiles/main.dir/flags.make
CMakeFiles/main.dir/src/pdensity.cpp.o: ../src/pdensity.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/rohan/Dropbox/Work/workspace/imseg/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/main.dir/src/pdensity.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/main.dir/src/pdensity.cpp.o -c /Users/rohan/Dropbox/Work/workspace/imseg/src/pdensity.cpp

CMakeFiles/main.dir/src/pdensity.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/main.dir/src/pdensity.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/rohan/Dropbox/Work/workspace/imseg/src/pdensity.cpp > CMakeFiles/main.dir/src/pdensity.cpp.i

CMakeFiles/main.dir/src/pdensity.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/main.dir/src/pdensity.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/rohan/Dropbox/Work/workspace/imseg/src/pdensity.cpp -o CMakeFiles/main.dir/src/pdensity.cpp.s

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
main: /usr/local/lib/libopencv_videostab.2.4.13.dylib
main: /usr/local/lib/libopencv_ts.a
main: /usr/local/lib/libopencv_superres.2.4.13.dylib
main: /usr/local/lib/libopencv_stitching.2.4.13.dylib
main: /usr/local/lib/libopencv_contrib.2.4.13.dylib
main: /usr/local/lib/libopencv_nonfree.2.4.13.dylib
main: /usr/local/lib/libopencv_ocl.2.4.13.dylib
main: /usr/local/lib/libopencv_gpu.2.4.13.dylib
main: /usr/local/lib/libopencv_photo.2.4.13.dylib
main: /usr/local/lib/libopencv_objdetect.2.4.13.dylib
main: /usr/local/lib/libopencv_legacy.2.4.13.dylib
main: /usr/local/lib/libopencv_video.2.4.13.dylib
main: /usr/local/lib/libopencv_ml.2.4.13.dylib
main: /usr/local/lib/libopencv_calib3d.2.4.13.dylib
main: /usr/local/lib/libopencv_features2d.2.4.13.dylib
main: /usr/local/lib/libopencv_highgui.2.4.13.dylib
main: /usr/local/lib/libopencv_imgproc.2.4.13.dylib
main: /usr/local/lib/libopencv_flann.2.4.13.dylib
main: /usr/local/lib/libopencv_core.2.4.13.dylib
main: CMakeFiles/main.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/rohan/Dropbox/Work/workspace/imseg/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable main"
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
	cd /Users/rohan/Dropbox/Work/workspace/imseg/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/rohan/Dropbox/Work/workspace/imseg /Users/rohan/Dropbox/Work/workspace/imseg /Users/rohan/Dropbox/Work/workspace/imseg/build /Users/rohan/Dropbox/Work/workspace/imseg/build /Users/rohan/Dropbox/Work/workspace/imseg/build/CMakeFiles/main.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/main.dir/depend

