#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <chrono>

// Utility function to check for CUDA errors
void checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA Runtime Error: " << cudaGetErrorString(result) << std::endl;
        exit(-1);
    }
}

// Utility function to check for cuBLAS errors
void checkCublas(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS Error: " << status << std::endl; // You might want to add more descriptive error messages here
        exit(-1);
    }
}

// Function to allocate memory, copy data, and invoke cuBLAS
void matrixMultiplyCublas(float* h_A, float* h_C, int rows, int cols) {
    float *d_A, *d_C;
    size_t size_A = rows * cols * sizeof(float);
    size_t size_C = rows * rows * sizeof(float);

    // Allocate device memory
    checkCuda(cudaMalloc(&d_A, size_A));
    checkCuda(cudaMalloc(&d_C, size_C));

    // Copy input data from host to device
    checkCuda(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));

    // Create cuBLAS handle
    cublasHandle_t handle;
    checkCublas(cublasCreate(&handle)); // Use checkCublas here

    // Set up parameters for cuBLAS Sgemm (single-precision matrix multiplication)
    float alpha = 1.0f;
    float beta = 0.0f;

    // Perform matrix multiplication using cuBLAS Sgemm (A * A^T)
    checkCublas(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, rows, rows, cols,
                          &alpha, d_A, cols, d_A, cols, &beta, d_C, rows)); // Use checkCublas here

    // Copy result from device to host
    checkCuda(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));

    // Destroy cuBLAS handle
    checkCublas(cublasDestroy(handle)); // Use checkCublas here

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
    // Define matrix dimensions
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

    // Benchmark large matrix multiplication with cuBLAS
    auto start_large = std::chrono::high_resolution_clock::now();
    matrixMultiplyCublas(large_matrix, large_result, large_rows, large_cols);
    auto end_large = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_large = end_large - start_large;

    // Benchmark mid matrix multiplication with cuBLAS
    auto start_mid = std::chrono::high_resolution_clock::now();
    matrixMultiplyCublas(mid_matrix, mid_result, mid_rows, mid_cols);
    auto end_mid = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_mid = end_mid - start_mid;

    // Benchmark small matrix multiplication with cuBLAS
    auto start_small = std::chrono::high_resolution_clock::now();
    matrixMultiplyCublas(small_matrix, small_result, small_rows, small_cols);
    auto end_small = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_small = end_small - start_small;

    // Print results
    std::cout << "Large matrix multiplication time (cuBLAS): " << elapsed_large.count() << " seconds" << std::endl;
    std::cout << "Mid matrix multiplication time (cuBLAS): " << elapsed_mid.count() << " seconds" << std::endl;
    std::cout << "Small matrix multiplication time (cuBLAS): " << elapsed_small.count() << " seconds" << std::endl;

    // Clean up
    delete[] large_matrix;
    delete[] mid_matrix;
    delete[] small_matrix;
    delete[] large_result;
    delete[] mid_result;
    delete[] small_result;

    return 0;
}