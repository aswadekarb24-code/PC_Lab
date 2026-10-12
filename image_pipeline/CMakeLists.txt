cmake_minimum_required(VERSION 3.18)
project(ImagePipeline LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Add CUDA source file
add_executable(pipeline_week3 src/main.cpp src/kernels.cu)

target_link_libraries(pipeline_week3 PRIVATE 
    ${OpenCV_LIBS} 
    OpenMP::OpenMP_CXX
    CUDA::cudart
)

target_include_directories(pipeline_week3 PRIVATE ${OpenCV_INCLUDE_DIRS} ${CUDAToolkit_INCLUDE_DIRS})