cmake_minimum_required(VERSION 3.16)

project(imgdiff LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED)

add_executable(imgdiff
  src/imgdiff.cpp
)

target_link_libraries(imgdiff PRIVATE ${OpenCV_LIBS})

add_executable(gen_test_pair
  src/gen_test_pair.cpp
)

target_link_libraries(gen_test_pair PRIVATE ${OpenCV_LIBS})
