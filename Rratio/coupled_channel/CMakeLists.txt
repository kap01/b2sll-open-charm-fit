# - Main CMake buildscript for costheta
# Comments in a CMake Script are lines begining with a '#'

# - Set CMake requirements then declare project
cmake_minimum_required(VERSION 3.2)

# Declare project
project(RModel VERSION 0.1)

# - When Makefiles are generated, output all command lines by default
#   Do this to begin with so it's easy to see what compiler command/flags
#   are used. This can also be done by removing the 'set' command and
#   running make as 'make VERBOSE=1'.
set(CMAKE_VERBOSE_MAKEFILE ON)

# - Don't allow C++ Compiler Vendor Extensions
set(CMAKE_CXX_EXTENSIONS OFF)

# - Set C++ Standard directly to c++17
set(CMAKE_CXX_STANDARD 17)

# - Use our standard set of flags
set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wfatal-errors  -pedantic")

# - ROOT installation
list(APPEND CMAKE_PREFIX_PATH $ENV{ROOTSYS})

# - Find ROOT
find_package(ROOT COMPONENTS XMLIO MathCore MathMore Minuit Physics RooFit RooFitCore RooStats Minuit2)

# - Find BOOST
find_package(Boost)
if(Boost_FOUND)
	include_directories(${Boost_INCLUDE_DIRS})
endif()

# - Find GSL
find_package(GSL)
if(GSL_FOUND)
	include_directories(${GSL_INCLUDE_DIRS})
endif()


#SET(SOURCE_DIR "/src")
add_library(root_interface INTERFACE IMPORTED)
target_include_directories(root_interface INTERFACE ${ROOT_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS} ${GSL_INCLUDE_DIRS})
target_link_libraries(root_interface INTERFACE ${ROOT_LIBRARIES} ${Boost_LIBRARIES} ${GSL_LIBRARIES})

add_subdirectory(src)


