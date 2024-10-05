cmake_minimum_required(VERSION 3.10)

# Project name and set it to enable CUDA
project(CudaVectorAdd LANGUAGES CXX CUDA)

# Specify the CUDA standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11) 
find_package(Eigen3 REQUIRED)
find_package(CUDA REQUIRED) 

# Add the executable target (the .cu file is treated as the source file)
add_executable(minimal_cuda  src/minimal_cuda.cu)
set_target_properties(minimal_cuda PROPERTIES CUDA_ARCHITECTURES 61)

# Add executable target
add_executable(matrix_operations src/matrix_operations.cu)

# Set the compute capability (modify according to your GPU)
set_target_properties(matrix_operations PROPERTIES CUDA_ARCHITECTURES 61)

# Add an executable
add_executable(cublas_example src/cublas_example.cu)
target_link_libraries(cublas_example /usr/local/cuda/lib64/libcublas.so)
set_target_properties(cublas_example PROPERTIES CUDA_ARCHITECTURES 61)


