#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

__global__ void lorenz_kernel(float* u, float dt, int num_steps) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  // Each thread handles one trajectory
  if (i < num_steps) {
    float x = u[i * 3];
    float y = u[i * 3 + 1];
    float z = u[i * 3 + 2];

    for (int step = 0; step < num_steps; ++step) {
      // Lorenz equations
      float dx = 10.0f * (y - x);
      float dy = x * (28.0f - z) - y;
      float dz = x * y - (8.0f / 3.0f) * z;

      // Simple Euler integration (replace with RK4 for better accuracy)
      x += dt * dx;
      y += dt * dy;
      z += dt * dz;
    }

    u[i * 3] = x;
    u[i * 3 + 1] = y;
    u[i * 3 + 2] = z;
  }
}

int main() {
  // Parameters
  float dt = 0.01f;
  int num_steps = 1000;
  int num_trajectories = 1; // Example: simulate many trajectories

  // Allocate host memory
  float* h_u = new float[num_trajectories * 3];

  // Initialize initial conditions (example: random values)
  for (int i = 0; i < num_trajectories * 3; ++i) {
    h_u[i] = rand() / (float)RAND_MAX;
  }

  // Allocate device memory
  float* d_u;
  cudaMalloc(&d_u, num_trajectories * 3 * sizeof(float));

  // Copy data to device
  cudaMemcpy(d_u, h_u, num_trajectories * 3 * sizeof(float),
             cudaMemcpyHostToDevice);

  // Launch kernel with 1 block and num_trajectories threads
  int threadsPerBlock = 256; // Adjust as needed
  int blocksPerGrid =
      (num_trajectories + threadsPerBlock - 1) / threadsPerBlock;

  // Start timer
  auto start = std::chrono::high_resolution_clock::now();

  lorenz_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_u, dt, num_steps);

  // Wait for kernel to finish
  cudaDeviceSynchronize();

  // Stop timer
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;

  // Copy data back to host
  cudaMemcpy(h_u, d_u, num_trajectories * 3 * sizeof(float),
             cudaMemcpyDeviceToHost);

  // Print timing results
  std::cout << "CUDA integration time: " << elapsed.count() << " seconds"
            << std::endl;

  // Process results (example: print final values of the first trajectory)
  std::cout << "Final values: " << h_u[0] << " " << h_u[1] << " " << h_u[2]
            << std::endl;

  // Free memory
  delete[] h_u;
  cudaFree(d_u);

  return 0;
}