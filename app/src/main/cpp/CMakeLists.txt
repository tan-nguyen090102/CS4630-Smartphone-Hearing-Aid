cmake_minimum_required(VERSION 3.22.1)

project("network")


add_library(
        ${CMAKE_PROJECT_NAME} SHARED network.cpp
)

include_directories(include/)

target_link_libraries(
        ${CMAKE_PROJECT_NAME}
        android
        log
)



find_library(
        ${CMAKE_PROJECT_NAME}
        log-lib
        log
)

