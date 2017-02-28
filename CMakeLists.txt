cmake_minimum_required(VERSION 3.6)
project(object_tracking)


find_package(OpenCV REQUIRED PATHS "/usr/local/Cellar/opencv3/3.2.0/share/OpenCV/")
set(CMAKE_CXX_STANDARD 14)
set(SOURCE_FILES tracking.cpp libs/libowi.h libs/libowi.cpp)
add_executable(object_tracking ${SOURCE_FILES})
target_link_libraries(object_tracking ${OpenCV_LIBS} usb-1.0)
