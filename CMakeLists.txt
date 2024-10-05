cmake_minimum_required(VERSION 3.10)

# Project name and set it to enable CUDA
project(CudaVectorAdd LANGUAGES CXX CUDA)

# Specify the CUDA standard
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)

# Add the executable target (the .cu file is treated as the source file)
add_executable(minimal_cuda  src/minimal_cuda.cu)

# Set the compute capability (optional, based on your GPU architecture)
# Example: Set for compute capability 6.1, change this for your GPU.
set_target_properties(minimal_cuda PROPERTIES
    CUDA_ARCHITECTURES 61
)

# Add executable target
add_executable(matrix_operations src/matrix_operations.cu)

# Set the compute capability (modify according to your GPU)
set_target_properties(matrix_operations PROPERTIES CUDA_ARCHITECTURES 61)

# Add an executable
add_executable(cublas_example src/cublas_example.cu)

target_link_libraries(cublas_example /usr/local/cuda/lib64/libcublas.so)

# Set CUDA architecture (e.g., sm_75 for Turing GPUs)
# set_target_properties(cublas_example PROPERTIES CUDA_ARCHITECTURES "sm_75")
# set_target_properties(cublas_example PROPERTIES CUDA_ARCHITECTURES "sm_60;sm_70;sm_75")

# Link with cuBLAS
# target_link_libraries(cublas_example CUDA::cublas)


