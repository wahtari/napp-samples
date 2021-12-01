CMAKE_MINIMUM_REQUIRED(VERSION 2.8.12)
PROJECT("Wahtari Person Detection C++ Raw Sample")

# OpenCV
FIND_PACKAGE(OpenCV REQUIRED)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})

# Wahtari nCam Controller Lib.
set(nCamCtrl_C_DIR controller-libs/c)
set(nCamCtrl_CPP_DIR controller-libs/cpp)
set(nCamCtrl_LIBS nlab-ctrl nlab-ctrl-cpp)
INCLUDE_DIRECTORIES(${nCamCtrl_C_DIR} ${nCamCtrl_CPP_DIR})
LINK_DIRECTORIES(${nCamCtrl_CPP_DIR})

# Vimba
set(Vimba_LIBS VimbaC VimbaCPP)
INCLUDE_DIRECTORIES("/usr/local/include/VimbaCPP/Include")

ADD_EXECUTABLE(sample sample.cpp)
TARGET_LINK_LIBRARIES(sample pthread ${OpenCV_LIBS} ${nCamCtrl_LIBS} ${Vimba_LIBS})
