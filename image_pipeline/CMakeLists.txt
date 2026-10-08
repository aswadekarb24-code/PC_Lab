cmake_minimum_required(VERSION 3.18)
project(ImagePipeline LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

add_executable(pipeline_week1 src/main.cpp)

target_link_libraries(pipeline_week1 PRIVATE 
    ${OpenCV_LIBS} 
    OpenMP::OpenMP_CXX
    CUDA::cudart
)

target_include_directories(pipeline_week1 PRIVATE ${OpenCV_INCLUDE_DIRS})