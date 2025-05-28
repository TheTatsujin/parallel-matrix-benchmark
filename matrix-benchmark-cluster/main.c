#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define ERROR_TOO_FEW_THREADS 0x12;
#define ERROR_TOO_MANY_THREADS 0x54
#define ERROR_MEMORY_ALLOCATION 0x66

#define MASTER_NODE 0
#define SLAVE_NODE 1

#define THREAD_NUMBER 6
#define NODE_NUMBER 2

#define MATRIX_SIZE 500

#define PI 3.14159265358979323846

int matrix_allocate(const int size, float** A) {
  *A = (float*)malloc(size * size * sizeof(float));
  if (*A == NULL) return ERROR_MEMORY_ALLOCATION;
  return 0;
}

int matrix_allocate_half(const int half_size, float** A) {
  *A = (float*)malloc(half_size * MATRIX_SIZE * sizeof(float));
  if (*A == NULL) return ERROR_MEMORY_ALLOCATION;
  return 0;
}

int matrix_init
(const int size, float** A)
{
  if (matrix_allocate(size, A) != 0) return ERROR_MEMORY_ALLOCATION;
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < MATRIX_SIZE; j++) (*A)[i*MATRIX_SIZE + j] = PI * (float) i + j;
  }
  return 0;
}


int matrix_init_half
(const int half_size, float** A)
{
  if (matrix_allocate_half(half_size, A) != 0) return ERROR_MEMORY_ALLOCATION;
  for (int i = 0; i < half_size; i++) {
    for (int j = 0; j < MATRIX_SIZE; j++) (*A)[i*MATRIX_SIZE + j] = PI * (float) i + j;
  }
  return 0;
}


int matrix_parallel_mult_half
(const int half_size, const int thread_number, const float* A, const float* B, float* result)
{
  if (thread_number > half_size) return ERROR_TOO_MANY_THREADS;
  if (thread_number < 2) return ERROR_TOO_FEW_THREADS;

  int i, j;
  #pragma omp for private(i,j) collapse(2) num_threads(thread_number)
    for (i = 0; i < half_size; i++) {
      for (j = 0; j < MATRIX_SIZE; j++) {
        float temp = (float) 0.;
        for (int k = 0; k < MATRIX_SIZE; k++) temp += A[i*MATRIX_SIZE + k] * B[j*MATRIX_SIZE + k];
        result[i*MATRIX_SIZE + j] = temp;
      }
    }


  return 0;
}

void matrix_print(const int rows, const int columns, const float* A) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < columns; j++) {
      if (i*columns + j == MATRIX_SIZE) printf("HELLO ----------\n");
      printf("%f ", A[i*columns + j]);

    }
    printf("\n");
  }
}



int main(int argc, char** argv) {
  float *A, *B, *result;
  float *full_result;
  const int master_matrix_size = MATRIX_SIZE / 2;
  const int slave_matrix_size = MATRIX_SIZE - master_matrix_size;
  int my_half_size;

  MPI_Init(&argc, &argv);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == MASTER_NODE) {
    my_half_size = master_matrix_size;
    matrix_init(MATRIX_SIZE, &B);
    MPI_Send(B, MATRIX_SIZE * MATRIX_SIZE, MPI_FLOAT, SLAVE_NODE, 0, MPI_COMM_WORLD);
  }
  else {
    my_half_size = slave_matrix_size;
    matrix_allocate(MATRIX_SIZE, &B);
    MPI_Recv(B, MATRIX_SIZE * MATRIX_SIZE, MPI_FLOAT, MASTER_NODE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  matrix_init_half(my_half_size, &A);

  matrix_allocate_half(my_half_size, &result);
  for (int i = 0; i < my_half_size; i++) {
    for (int j = 0; j < MATRIX_SIZE; j++) result[i*MATRIX_SIZE + j] = (float) 0.;
  }

  matrix_parallel_mult_half(my_half_size, THREAD_NUMBER, A, B, result);

  if (rank == MASTER_NODE) {
    matrix_allocate(MATRIX_SIZE, &full_result);
    MPI_Gather(result, my_half_size * MATRIX_SIZE, MPI_FLOAT, full_result, slave_matrix_size * MATRIX_SIZE, MPI_FLOAT, MASTER_NODE, MPI_COMM_WORLD);
    matrix_print(MATRIX_SIZE, MATRIX_SIZE, full_result);
    free(full_result);
  }
  else MPI_Gather(result, my_half_size * MATRIX_SIZE, MPI_FLOAT, NULL, 0, MPI_FLOAT, MASTER_NODE, MPI_COMM_WORLD);

  free(A);
  free(B);
  free(result);
  MPI_Finalize();

  return 0;
}