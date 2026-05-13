cmake_minimum_required(VERSION 3.16)

project(imgdiff LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED)

option(IMGDIFF_BUILD_GUI "Build Qt GUI (imgdiff_gui)" OFF)

add_executable(imgdiff
  src/imgdiff.cpp
)

target_link_libraries(imgdiff PRIVATE ${OpenCV_LIBS})

add_executable(gen_test_pair
  src/gen_test_pair.cpp
)

target_link_libraries(gen_test_pair PRIVATE ${OpenCV_LIBS})

if(IMGDIFF_BUILD_GUI)
  set(CMAKE_AUTOMOC ON)
  set(CMAKE_AUTORCC ON)
  set(CMAKE_AUTOUIC ON)

  find_package(Qt6 REQUIRED COMPONENTS Widgets)

  add_executable(imgdiff_gui
    src/imgdiff_gui.cpp
  )

  target_link_libraries(imgdiff_gui PRIVATE Qt6::Widgets)
endif()
