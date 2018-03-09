cmake_minimum_required(VERSION 3.8)
set(OpenCV_DIR )
project(recognise-your-own-face)
FIND_PACKAGE(OpenCV REQUIRED)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES face-rec/take_photos.cpp) #face-rec/train.cpp face-rec/take_photos.cpp face-rec/rec-your-own-face.cpp face-rec/test.cpp)
add_executable(recognise-your-own-face ${SOURCE_FILES})
target_link_libraries(recognise-your-own-face ${OpenCV_LIBS})


## cmake needs this line
#cmake_minimum_required(VERSION 2.8)
#
## Define project name
#project(opencvTest)
#
## Find OpenCV, you may need to set OpenCV_DIR variable
## to the absolute path to the directory containing OpenCVConfig.cmake file
## via the command line or GUI
#find_package(OpenCV REQUIRED)
#
## If the package has been found, several variables will
## be set, you can find the full list with descriptions
## in the OpenCVConfig.cmake file.
## Print some message showing some of them
#message(STATUS "OpenCV library status:")
#message(STATUS "    version: ${OpenCV_VERSION}")
#message(STATUS "    libraries: ${OpenCV_LIBS}")
#message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
#
## Add OpenCV headers location to your include paths
#include_directories(${OpenCV_INCLUDE_DIRS})
#
## Declare the executable target built from your sources
#add_executable(main main.cpp)
#
## Link your application with OpenCV libraries
#target_link_libraries(main ${OpenCV_LIBS})