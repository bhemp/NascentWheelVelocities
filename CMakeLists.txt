cmake_minimum_required(VERSION 3.4.5)
project(wheelvelocity)

set(SOURCE main.cpp)

set(CMAKE_CXX_FLAGS "-Wall -std=c++0x")

add_executable(${PROJECT_NAME} ${SOURCE})

# set( OpenCV_DIR, "~/opencv_3.4.5/release/build" )
set( OpenCV_FIND_QUIETLY 1)
SET(OpenCV_INSTALL_PATH /home/user/opencv-3.4.5/release/build)

set(OpenCV_INCLUDE_DIRS
  /usr/local/opencv-3.4.5/include/opencv2
  /usr/local/opencv_contrib-3.4.5/modules
)

set(OpenCV_LIBS opencv_core opencv_imgcodecs opencv_imgproc opencv_calib3d opencv_video opencv_features2d opencv_ml opencv_highgui opencv_objdetect opencv_tracking)

find_package(OpenCV REQUIRED PATHS
/usr/local/share/OpenCV NO_DEFAULT_PATH)
include_directories(${OpenCV_INCLUDE_DIRS})

# If the package has been found, several variables will
# be set, you can find the full list with descriptions
# in the OpenCVConfig.cmake file.
# Print some message showing some of them
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")


target_link_libraries(wheelvelocity ${OpenCV_LIBS})