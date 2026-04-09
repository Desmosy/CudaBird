cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(cudabird LANGUAGES CXX CUDA)

option(CUDABIRD_ENABLE_OPENCV "Enable optional OpenCV hooks" ON)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 80 86)
endif()

find_package(CUDAToolkit REQUIRED)

if(CUDABIRD_ENABLE_OPENCV)
    find_package(OpenCV QUIET)
endif()

set(CUDABIRD_CORE_SOURCES
    src/config.cpp
    src/cuda_utils.cu
    src/game/game_kernels.cu
    src/neural_net/network_kernels.cu
    src/evolution/evolution_kernels.cu
    src/visualization/renderer.cpp
)

add_library(cudabird_core STATIC ${CUDABIRD_CORE_SOURCES})

target_include_directories(cudabird_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(cudabird_core
    PUBLIC
        CUDA::cudart
        CUDA::curand
)

target_compile_options(cudabird_core
    PUBLIC
        $<$<COMPILE_LANGUAGE:CUDA>:-allow-unsupported-compiler>
)

if(OpenCV_FOUND)
    target_compile_definitions(cudabird_core PUBLIC CUDABIRD_HAVE_OPENCV=1)
    target_include_directories(cudabird_core PUBLIC ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(cudabird_core PUBLIC ${OpenCV_LIBS})
endif()

set_target_properties(cudabird_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

add_executable(cudabird src/main.cu)
target_link_libraries(cudabird PRIVATE cudabird_core)
set_target_properties(cudabird PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

enable_testing()

foreach(test_name IN ITEMS test_game test_network test_evolution)
    add_executable(${test_name} tests/${test_name}.cu)
    target_link_libraries(${test_name} PRIVATE cudabird_core)
    set_target_properties(${test_name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()

if(OpenCV_FOUND)
    add_executable(test_renderer tests/test_renderer.cpp)
    target_link_libraries(test_renderer PRIVATE cudabird_core)
    add_test(NAME test_renderer COMMAND test_renderer)
endif()
