# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.0

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

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /net/cremi/jarnault/espaces/travail/Projet/fichiers

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /net/cremi/jarnault/espaces/travail/Projet/fichiers/build

# Include any dependencies generated for this target.
include CMakeFiles/atoms.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/atoms.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/atoms.dir/flags.make

CMakeFiles/atoms.dir/src/main.c.o: CMakeFiles/atoms.dir/flags.make
CMakeFiles/atoms.dir/src/main.c.o: ../src/main.c
	$(CMAKE_COMMAND) -E cmake_progress_report /net/cremi/jarnault/espaces/travail/Projet/fichiers/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building C object CMakeFiles/atoms.dir/src/main.c.o"
	/usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -o CMakeFiles/atoms.dir/src/main.c.o   -c /net/cremi/jarnault/espaces/travail/Projet/fichiers/src/main.c

CMakeFiles/atoms.dir/src/main.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/atoms.dir/src/main.c.i"
	/usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -E /net/cremi/jarnault/espaces/travail/Projet/fichiers/src/main.c > CMakeFiles/atoms.dir/src/main.c.i

CMakeFiles/atoms.dir/src/main.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/atoms.dir/src/main.c.s"
	/usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -S /net/cremi/jarnault/espaces/travail/Projet/fichiers/src/main.c -o CMakeFiles/atoms.dir/src/main.c.s

CMakeFiles/atoms.dir/src/main.c.o.requires:
.PHONY : CMakeFiles/atoms.dir/src/main.c.o.requires

CMakeFiles/atoms.dir/src/main.c.o.provides: CMakeFiles/atoms.dir/src/main.c.o.requires
	$(MAKE) -f CMakeFiles/atoms.dir/build.make CMakeFiles/atoms.dir/src/main.c.o.provides.build
.PHONY : CMakeFiles/atoms.dir/src/main.c.o.provides

CMakeFiles/atoms.dir/src/main.c.o.provides.build: CMakeFiles/atoms.dir/src/main.c.o

CMakeFiles/atoms.dir/src/tools.c.o: CMakeFiles/atoms.dir/flags.make
CMakeFiles/atoms.dir/src/tools.c.o: ../src/tools.c
	$(CMAKE_COMMAND) -E cmake_progress_report /net/cremi/jarnault/espaces/travail/Projet/fichiers/build/CMakeFiles $(CMAKE_PROGRESS_2)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building C object CMakeFiles/atoms.dir/src/tools.c.o"
	/usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -o CMakeFiles/atoms.dir/src/tools.c.o   -c /net/cremi/jarnault/espaces/travail/Projet/fichiers/src/tools.c

CMakeFiles/atoms.dir/src/tools.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/atoms.dir/src/tools.c.i"
	/usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -E /net/cremi/jarnault/espaces/travail/Projet/fichiers/src/tools.c > CMakeFiles/atoms.dir/src/tools.c.i

CMakeFiles/atoms.dir/src/tools.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/atoms.dir/src/tools.c.s"
	/usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -S /net/cremi/jarnault/espaces/travail/Projet/fichiers/src/tools.c -o CMakeFiles/atoms.dir/src/tools.c.s

CMakeFiles/atoms.dir/src/tools.c.o.requires:
.PHONY : CMakeFiles/atoms.dir/src/tools.c.o.requires

CMakeFiles/atoms.dir/src/tools.c.o.provides: CMakeFiles/atoms.dir/src/tools.c.o.requires
	$(MAKE) -f CMakeFiles/atoms.dir/build.make CMakeFiles/atoms.dir/src/tools.c.o.provides.build
.PHONY : CMakeFiles/atoms.dir/src/tools.c.o.provides

CMakeFiles/atoms.dir/src/tools.c.o.provides.build: CMakeFiles/atoms.dir/src/tools.c.o

# Object files for target atoms
atoms_OBJECTS = \
"CMakeFiles/atoms.dir/src/main.c.o" \
"CMakeFiles/atoms.dir/src/tools.c.o"

# External object files for target atoms
atoms_EXTERNAL_OBJECTS =

../bin/atoms: CMakeFiles/atoms.dir/src/main.c.o
../bin/atoms: CMakeFiles/atoms.dir/src/tools.c.o
../bin/atoms: CMakeFiles/atoms.dir/build.make
../bin/atoms: libsotl/libsotl.so
../bin/atoms: /usr/lib/x86_64-linux-gnu/libOpenCL.so
../bin/atoms: /usr/lib/x86_64-linux-gnu/libm.so
../bin/atoms: /usr/lib/x86_64-linux-gnu/libGLU.so
../bin/atoms: /usr/lib/x86_64-linux-gnu/libGL.so
../bin/atoms: /usr/lib/x86_64-linux-gnu/libSM.so
../bin/atoms: /usr/lib/x86_64-linux-gnu/libICE.so
../bin/atoms: /usr/lib/x86_64-linux-gnu/libX11.so
../bin/atoms: /usr/lib/x86_64-linux-gnu/libXext.so
../bin/atoms: /usr/lib/x86_64-linux-gnu/libglut.so
../bin/atoms: /usr/lib/x86_64-linux-gnu/libXmu.so
../bin/atoms: /usr/lib/x86_64-linux-gnu/libXi.so
../bin/atoms: CMakeFiles/atoms.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking C executable ../bin/atoms"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/atoms.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/atoms.dir/build: ../bin/atoms
.PHONY : CMakeFiles/atoms.dir/build

CMakeFiles/atoms.dir/requires: CMakeFiles/atoms.dir/src/main.c.o.requires
CMakeFiles/atoms.dir/requires: CMakeFiles/atoms.dir/src/tools.c.o.requires
.PHONY : CMakeFiles/atoms.dir/requires

CMakeFiles/atoms.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/atoms.dir/cmake_clean.cmake
.PHONY : CMakeFiles/atoms.dir/clean

CMakeFiles/atoms.dir/depend:
	cd /net/cremi/jarnault/espaces/travail/Projet/fichiers/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /net/cremi/jarnault/espaces/travail/Projet/fichiers /net/cremi/jarnault/espaces/travail/Projet/fichiers /net/cremi/jarnault/espaces/travail/Projet/fichiers/build /net/cremi/jarnault/espaces/travail/Projet/fichiers/build /net/cremi/jarnault/espaces/travail/Projet/fichiers/build/CMakeFiles/atoms.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/atoms.dir/depend

