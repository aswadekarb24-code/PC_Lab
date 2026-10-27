cmake_minimum_required(VERSION 3.18)
project(ImagePipeline LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

# Optimised build by default so serial/parallel timings are representative
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Parallel pipeline: OpenMP + CUDA
add_executable(pipeline_parallel src/main.cpp src/kernels.cu)

target_link_libraries(pipeline_parallel PRIVATE
    ${OpenCV_LIBS}
    OpenMP::OpenMP_CXX
    CUDA::cudart
)

target_include_directories(pipeline_parallel PRIVATE src ${OpenCV_INCLUDE_DIRS} ${CUDAToolkit_INCLUDE_DIRS})

# Serial pipeline: plain single-threaded CPU (no OpenMP, no CUDA)
add_executable(pipeline_serial src/main_serial.cpp)

target_link_libraries(pipeline_serial PRIVATE ${OpenCV_LIBS})

target_include_directories(pipeline_serial PRIVATE src ${OpenCV_INCLUDE_DIRS})

# Encoder scaling benchmark: independent concurrent encodes (OpenCV only, no CUDA)
add_executable(encode_bench src/encode_bench.cpp)
target_link_libraries(encode_bench PRIVATE ${OpenCV_LIBS})
target_include_directories(encode_bench PRIVATE ${OpenCV_INCLUDE_DIRS})
