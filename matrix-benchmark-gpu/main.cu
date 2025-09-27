#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define ERROR_MEMORY_ALLOCATION 0x66

#define MATRIX_SIZE 1200
#define PI 3.14159265358979323846


void matrix_allocate
(const int size, float*** A) {
  *A = (float**)malloc(size * sizeof(float*));
  if (*A == NULL) return;
  for (int i = 0; i < size; i++) {
    (*A)[i] = (float*)malloc(size * sizeof(float));
    if ((*A)[i] == NULL) {
      for (int j = 0; j < i; j++) free((*A)[j]);
      free(*A);
      *A = NULL;
      return;
    }
  }
}

void matrix_free
(const int size, float*** A, float*** B, float*** result) {
  for (int i = 0; i < size; i++) {
    free((*A)[i]);
    free((*B)[i]);
    free((*result)[i]);
  }
  free(*A);
  free(*B);
  free(*result);
  *A = NULL;
  *B = NULL;
  *result = NULL;
}


int matrix_init
(const int size, float*** A, float*** B, float*** result)
{
  matrix_allocate(size, A);
  if (*A == NULL) return ERROR_MEMORY_ALLOCATION;

  matrix_allocate(size, B);
  if (*B == NULL) return ERROR_MEMORY_ALLOCATION;

  matrix_allocate(size, result);
  if (*result == NULL) return ERROR_MEMORY_ALLOCATION;

  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      (*A)[i][j] = PI * (float) i + j;
      (*B)[j][i] = (*A)[i][j];
      (*result)[i][j] = (float) 0.;
    }
  }

  return 0;
}

void matrix_transpose
(const int size, float** A)
{
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) A[i][j] = A[j][i];
  }
}

void matrix_print(const int rows, const int columns, float** A) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < columns; j++) {
      printf("%f ", A[i][j]);
    }
    printf("\n");
  }
}




void CUDA_matrix_copy(int size, float*** A, float*** A_gpu_copy) {
  cudaMalloc((void **) A_gpu_copy, size * sizeof(float*));

  for (int i = 0; i < size; i++) {
    cudaMalloc((void **) &((*A_gpu_copy)[i]), size * sizeof(float));
    
    cudaMemcpy((*A_gpu_copy)[i], (*A)[i], size * sizeof(float), cudaMemcpyHostToDevice);
  }
}

int main() {
  float **A, **B, **result;
  float ***A_gpu_copy, ***B_gpu_copy, ***result_gpu_copy;

  matrix_init(MATRIX_SIZE, &A, &B, &result);
  CUDA_matrix_copy(MATRIX_SIZE, &A, A_gpu_copy);
  CUDA_matrix_copy(MATRIX_SIZE, &B, B_gpu_copy);
  CUDA_matrix_copy(MATRIX_SIZE, &result, result_gpu_copy);

  return 0;
}
