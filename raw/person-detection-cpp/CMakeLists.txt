CMAKE_MINIMUM_REQUIRED(VERSION 2.8.12)
PROJECT("Wahtari Person Detection C++ Raw Sample")

FIND_PACKAGE(OpenCV REQUIRED)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})

LINK_DIRECTORIES(/usr/local/lib /work/controller-libs/cpp)
ADD_EXECUTABLE(sample sample.cpp MJPEGStreamer.hpp controller-libs/cpp/libnlab-ctrl.hpp)
TARGET_LINK_LIBRARIES(sample ${OpenCV_LIBS})
TARGET_LINK_LIBRARIES(sample VimbaC VimbaCPP)
TARGET_LINK_LIBRARIES(sample /lib/x86_64-linux-gnu/libpthread.so.0)
TARGET_LINK_LIBRARIES(sample controller-libs/cpp/libnlab-ctrl.so)
TARGET_LINK_LIBRARIES(sample controller-libs/cpp/libnlab-ctrl-cpp.so)
