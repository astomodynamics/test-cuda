#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

// CUDA kernel for matrix multiplication (A * A^T)
__global__ void matrixMultiplyKernel(float* A, float* C, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < rows) {
        float value = 0.0f;
        for (int i = 0; i < cols; i++) {
            value += A[row * cols + i] * A[col * cols + i];  // A * A^T
        }
        C[row * rows + col] = value;
    }
}

// Utility function to check for CUDA errors
void checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA Runtime Error: " << cudaGetErrorString(result) << std::endl;
        exit(-1);
    }
}

// Function to allocate memory, copy data, and invoke kernel
void matrixMultiply(float* h_A, float* h_C, int rows, int cols) {
    float *d_A, *d_C;
    size_t size_A = rows * cols * sizeof(float);
    size_t size_C = rows * rows * sizeof(float);

    // Allocate device memory
    checkCuda(cudaMalloc(&d_A, size_A));
    checkCuda(cudaMalloc(&d_C, size_C));

    // Copy input data from host to device
    checkCuda(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));

    // Set up block and grid dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((rows + blockDim.x - 1) / blockDim.x, (rows + blockDim.y - 1) / blockDim.y);

    // Launch matrix multiplication kernel
    matrixMultiplyKernel<<<gridDim, blockDim>>>(d_A, d_C, rows, cols);

    // Check for errors during kernel execution
    checkCuda(cudaPeekAtLastError());

    // Copy result from device to host
    checkCuda(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));

    // Free device memory
    checkCuda(cudaFree(d_A));
    checkCuda(cudaFree(d_C));
}

// Helper function to initialize matrix with random values
void randomMatrix(float* mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

int main() {
    // Define large, mid and small matrix dimensions
    int large_rows = 1000;
    int large_cols = 1000;
    int mid_rows = 100;
    int mid_cols = 100;
    int small_rows = 10;
    int small_cols = 10;

    // Allocate host memory
    float* large_matrix = new float[large_rows * large_cols];
    float* mid_matrix = new float[mid_rows * mid_cols];
    float* small_matrix = new float[small_rows * small_cols];
    float* large_result = new float[large_rows * large_rows];
    float* mid_result = new float[mid_rows * mid_rows];
    float* small_result = new float[small_rows * small_rows];

    // Initialize matrices with random values
    randomMatrix(large_matrix, large_rows, large_cols);
    randomMatrix(mid_matrix, mid_rows, mid_cols);
    randomMatrix(small_matrix, small_rows, small_cols);

    // Benchmark large matrix multiplication
    auto start_large = std::chrono::high_resolution_clock::now();
    matrixMultiply(large_matrix, large_result, large_rows, large_cols);
    auto end_large = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_large = end_large - start_large;

    // Benchmark mid matrix multiplication
    auto start_mid = std::chrono::high_resolution_clock::now();
    matrixMultiply(mid_matrix, mid_result, mid_rows, mid_cols);
    auto end_mid = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_mid = end_mid - start_mid;

    // Benchmark small matrix multiplication
    auto start_small = std::chrono::high_resolution_clock::now();
    matrixMultiply(small_matrix, small_result, small_rows, small_cols);
    auto end_small = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_small = end_small - start_small;

    // Print results
    std::cout << "Large matrix multiplication time: " << elapsed_large.count() << " seconds" << std::endl;
    std::cout << "Mid matrix multiplication time: " << elapsed_mid.count() << " seconds" << std::endl;
    std::cout << "Small matrix multiplication time: " << elapsed_small.count() << " seconds" << std::endl;

    // Clean up
    delete[] large_matrix;
    delete[] mid_matrix;
    delete[] small_matrix;
    delete[] large_result;
    delete[] mid_result;
    delete[] small_result;

    return 0;
}
