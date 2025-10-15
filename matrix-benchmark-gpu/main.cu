#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define ERROR_MEMORY_ALLOCATION 0x66
#define BLOCK_SIZE 16

#define MATRIX_SIZE 10000
#define PI 3.14159265358979323846


int matrix_allocate
(const int size, float** A) {
  *A = (float*)malloc(size * size * sizeof(float));
  if (*A == NULL) return ERROR_MEMORY_ALLOCATION;
  return 0;
}

int matrix_init
(const int size, float* A, float* B, float* result)
{
  if (A == NULL || B == NULL || result == NULL)
    return ERROR_MEMORY_ALLOCATION;
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      A[i*size + j] = PI * i + j;
      B[i*size + j] = PI * j + i;
      result[i*size + j] = 0.0;
    }
  }

  return 0;
}

void matrix_transpose
(const int size, float* A)
{
  for (int i = 0; i < size; i++) {
    for (int j = i + 1; j < size; j++) {
      float temp = A[i*size + j];
      A[i*size + j] = A[j*size + i];
      A[j*size + i] = temp;
    }
  }
}

void matrix_print(const int rows, const int columns, float* A) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < columns; j++) {
      printf("%.2f ", A[i*columns + j]);
    }
    printf("\n");
  }
}


int CUDA_matrix_copy(const int size, float* A, float** A_gpu_copy) {
  size_t type_size = size * size * sizeof(float);
  cudaMalloc(A_gpu_copy, type_size);
  cudaMemcpy(*A_gpu_copy, A, type_size, cudaMemcpyHostToDevice);
  return 0;
}


__global__ void CUDA_matrix_mult(const int size, float* A, float* B, float* result) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < size && col < size) {
    float sum = 0.0;
    for (int k = 0; k < size; k++) {
      sum += A[row * size + k] * B[col * size + k];
    }
    result[row * size + col] = sum;
  }
}


int main() {
  float *A, *B, *result;
  float *A_gpu_copy, *B_gpu_copy, *result_gpu_copy;

  if (matrix_allocate(MATRIX_SIZE, &A) == ERROR_MEMORY_ALLOCATION)
    return ERROR_MEMORY_ALLOCATION;
  if (matrix_allocate(MATRIX_SIZE, &B) == ERROR_MEMORY_ALLOCATION)
    return ERROR_MEMORY_ALLOCATION;
  if (matrix_allocate(MATRIX_SIZE, &result) == ERROR_MEMORY_ALLOCATION)
    return ERROR_MEMORY_ALLOCATION;

  if (matrix_init(MATRIX_SIZE, A, B, result) == ERROR_MEMORY_ALLOCATION)
    return ERROR_MEMORY_ALLOCATION;

  CUDA_matrix_copy(MATRIX_SIZE, A, &A_gpu_copy);

  matrix_transpose(MATRIX_SIZE, B);
  CUDA_matrix_copy(MATRIX_SIZE, B, &B_gpu_copy);
  CUDA_matrix_copy(MATRIX_SIZE, result, &result_gpu_copy);


  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
  dim3 gridDim((MATRIX_SIZE + blockDim.x - 1)/blockDim.x,
    (MATRIX_SIZE + blockDim.y - 1)/blockDim.y);

  CUDA_matrix_mult<<<gridDim, blockDim>>>(MATRIX_SIZE, A_gpu_copy, B_gpu_copy, result_gpu_copy);
  cudaDeviceSynchronize();
  cudaMemcpy(result, result_gpu_copy, MATRIX_SIZE * MATRIX_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(A_gpu_copy);
  cudaFree(B_gpu_copy);
  cudaFree(result_gpu_copy);

  free(A);
  free(B);
  free(result);

  return 0;
}
