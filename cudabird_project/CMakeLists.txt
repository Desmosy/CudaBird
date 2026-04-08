
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

include(FetchContent)
include(ExternalProject)

# Allow the user to specify a desired compiler version and install path
set(USER_SPECIFIED_C_COMPILER "" CACHE STRING "User-specified C compiler")
set(USER_SPECIFIED_CXX_COMPILER "" CACHE STRING "User-specified C++ compiler")
set(GPUTK_INSTALL_PATH "${CMAKE_CURRENT_SOURCE_DIR}/third_party/libgputk" CACHE PATH "Installation path for libgputk")

# Set compilers before project declaration
if(NOT "${USER_SPECIFIED_CXX_COMPILER}" STREQUAL "")
    set(CMAKE_CXX_COMPILER "${USER_SPECIFIED_CXX_COMPILER}")
endif()

if(NOT "${USER_SPECIFIED_C_COMPILER}" STREQUAL "")
    set(CMAKE_C_COMPILER "${USER_SPECIFIED_C_COMPILER}")
endif()

# Set CUDA flags before project declaration
set(CMAKE_CUDA_COMPILER_FLAGS "-allow-unsupported-compiler")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")
if(NOT "${USER_SPECIFIED_C_COMPILER}" STREQUAL "")
    set(CMAKE_CUDA_HOST_COMPILER "${USER_SPECIFIED_C_COMPILER}")
endif()

project(cudabird LANGUAGES CXX CUDA)

# C++ and CUDA standards
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDA REQUIRED)

# Find OpenCV (Useful for visualization/rendering later)
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

# Function to find libgputk
function(find_libgputk)
    # First check user-specified path
    if(NOT "${SYSTEM_GPUTK_PATH}" STREQUAL "")
        if(EXISTS "${SYSTEM_GPUTK_PATH}/include/gputk.h")
            set(GPUTK_FOUND TRUE PARENT_SCOPE)
            set(GPUTK_INCLUDE_DIR "${SYSTEM_GPUTK_PATH}/include" PARENT_SCOPE)
            set(GPUTK_LIBRARY_DIR "${SYSTEM_GPUTK_PATH}/lib" PARENT_SCOPE)
            message(STATUS "Found libgputk in specified path: ${SYSTEM_GPUTK_PATH}")
            return()
        endif()
    endif()

    # Check common system paths
    find_path(GPUTK_INCLUDE_DIR gputk.h
        PATHS
        /usr/include
        /usr/local/include
        /opt/local/include
        /usr/local/libgputk/include
    )

    find_library(GPUTK_LIBRARY
        NAMES gputk libgputk
        PATHS
        /usr/lib
        /usr/local/lib
        /opt/local/lib
        /usr/local/libgputk/lib
    )

    if(GPUTK_INCLUDE_DIR AND GPUTK_LIBRARY)
        get_filename_component(GPUTK_LIBRARY_DIR ${GPUTK_LIBRARY} DIRECTORY)
        set(GPUTK_FOUND TRUE PARENT_SCOPE)
        set(GPUTK_INCLUDE_DIR ${GPUTK_INCLUDE_DIR} PARENT_SCOPE)
        set(GPUTK_LIBRARY_DIR ${GPUTK_LIBRARY_DIR} PARENT_SCOPE)
        find_library(GPUTK_LIB_PATH gputk REQUIRED)
        add_library(libgputk SHARED IMPORTED)
        set_target_properties(libgputk PROPERTIES IMPORTED_LOCATION ${GPUTK_LIB_PATH})
        message(STATUS "Found system libgputk:")
        message(STATUS "    include: ${GPUTK_INCLUDE_DIR}")
        message(STATUS "    library: ${GPUTK_LIBRARY_DIR}")
    else()
        set(GPUTK_FOUND FALSE PARENT_SCOPE)
        message(STATUS "System libgputk not found, will build from source")
    endif()
endfunction()

# Try to find libgputk
find_libgputk()

# Build libgputk if not found
if(NOT GPUTK_FOUND)
    ExternalProject_Add(libgputk
        GIT_REPOSITORY https://github.com/ajdillhoff/libgputk.git
        GIT_TAG main
        CMAKE_ARGS
            -DCMAKE_INSTALL_PREFIX=${GPUTK_INSTALL_PATH}
            -DBUILD_SHARED_LIBS=ON
        BUILD_ALWAYS OFF
        TEST_COMMAND ""
    )
    set(GPUTK_INCLUDE_DIR "${GPUTK_INSTALL_PATH}/include")
    set(GPUTK_LIBRARY_DIR "${GPUTK_INSTALL_PATH}/lib")
endif()

# Additional CUDA flags
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
    -allow-unsupported-compiler
)

# ---- Source files ----
set(CUDA_SOURCES
    src/main.cu
    src/cuda_utils.cu
    src/game/game_kernels.cu
    src/neural_net/network_kernels.cu
    src/evolution/evolution_kernels.cu
)

set(CPP_SOURCES
    src/config.cpp
    src/visualization/renderer.cpp
)

add_executable(cudabird ${CUDA_SOURCES} ${CPP_SOURCES})

target_include_directories(cudabird PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${GPUTK_INSTALL_PATH}
    ${GPUTK_INSTALL_PATH}/libgputk/include
    ${GPUTK_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(cudabird PRIVATE
    curand
    gputk
    m
    stdc++
    ${OpenCV_LIBS}
)
if(NOT GPUTK_FOUND)
    add_dependencies(cudabird libgputk)
endif()

set_target_properties(cudabird PROPERTIES
    CUDA_ARCHITECTURES "70;80;86"
)

