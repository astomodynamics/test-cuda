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
