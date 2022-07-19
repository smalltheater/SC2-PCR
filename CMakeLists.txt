cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(SC2PCR)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g -pthread")

find_package(Torch REQUIRED)
find_package(yaml-cpp REQUIRED )
#find_package(ZLIB REQUIRED)
find_package(PCL REQUIRED QUIET)
#find_package(OpenMP REQUIRED)


include_directories(include
        ${YAML_CPP_INCLUDE_DIR}
        #${ZLIB_INCLUDE_DIRS}
        ${PCL_INCLUDE_DIRS})
link_directories(  ${PCL_LIBRARY_DIRS})

set(cnpy "/home/yunqi/snap/tools/cnpy/release/lib")

#
#add_library(cnpy SHARED "src/cnpy.cpp")
#target_link_libraries(cnpy ${ZLIB_LIBRARIES})
#install(TARGETS "cnpy" LIBRARY DESTINATION lib PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)

add_executable(dcgan src/main.cpp src/SC2PCR.cpp  src/YamlServer.cpp src/utility.cpp)
#target_compile_options(dcgan PRIVATE ${OpenMP_CXX_FLAGS})
target_link_libraries(visualize
        ${TORCH_LIBRARIES}
        yaml-cpp
       # cnpy
        ${PCL_LIBRARIES} )

