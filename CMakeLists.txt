cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(SC2PCR)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g -pthread")

find_package(yaml-cpp REQUIRED )
#find_package(ZLIB REQUIRED)
find_package(PCL REQUIRED QUIET)
find_package(OpenMP REQUIRED)

##cuda and libtorch##
## path should be modified
set(CMAKE_PREFIX_PATH /home/yunqi/libtorch/)
include_directories(/home/yunqi/libtorch/include)
include_directories(/home/yunqi/libtorch/include/torch/csrc/api/include)
include_directories(/usr/local/cuda/include)
link_directories(/home/yunqi/libtorch)
##

find_package(Torch REQUIRED)

include_directories(include
        ${YAML_CPP_INCLUDE_DIR}
        #${ZLIB_INCLUDE_DIRS}
        ${PCL_INCLUDE_DIRS})
link_directories(  ${PCL_LIBRARY_DIRS})


add_executable(visualize src/main.cpp src/SC2PCR.cpp  src/YamlServer.cpp src/utility.cpp)
target_compile_options(visualize PRIVATE ${OpenMP_CXX_FLAGS})
target_link_libraries(visualize
        ${TORCH_LIBRARIES}
        yaml-cpp
       # cnpy
        ${OpenMP_CXX_FLAGS}
        ${PCL_LIBRARIES} )

