cmake_minimum_required(VERSION 3.5)

set(PROJECT_NAME ann_project)
project(${PROJECT_NAME} CXX)

find_package(Threads REQUIRED)

if (NOT CMAKE_BUILD_TYPE) 
  set(CMAKE_BUILD_TYPE Release) 
endif() 

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11")

include_directories(include)

file(GLOB SOURCES_LIB "${PROJECT_SOURCE_DIR}/src/lib/*.cpp")

add_library(ann_lib ${SOURCES_LIB})

file(GLOB SOURCES "${PROJECT_SOURCE_DIR}/src/*.cpp")

add_executable(${PROJECT_NAME} ${SOURCES})

target_link_libraries(${PROJECT_NAME} ann_lib)

option(BUILD_TESTS "Build the tests" OFF)

if(BUILD_TESTS)

  add_subdirectory(libs/googletest)

  enable_testing()

  set(PROJECT_TEST_NAME ${PROJECT_NAME}_test)

  include_directories(${GTEST_INCLUDE_DIRS})

  file(GLOB TEST_SRC_FILES "${PROJECT_SOURCE_DIR}/tests/*.cpp")
  add_executable(${PROJECT_TEST_NAME} ${TEST_SRC_FILES})

  target_link_libraries(${PROJECT_TEST_NAME} ann_lib)

  add_dependencies(${PROJECT_TEST_NAME} googletest)

  target_link_libraries(${PROJECT_TEST_NAME}
    ${GTEST_LIBS_DIR}/libgtest.a
    ${GTEST_LIBS_DIR}/libgtest_main.a
  )

  target_link_libraries(${PROJECT_TEST_NAME} ${CMAKE_THREAD_LIBS_INIT})

  add_test(NAME project_test COMMAND ${PROJECT_TEST_NAME})

endif()