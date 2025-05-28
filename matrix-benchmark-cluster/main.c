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

int matrix_init
(const int size, float** A, const int start, const int end)
{
  if (matrix_allocate(size, A) != 0) return ERROR_MEMORY_ALLOCATION;
  for (int i = start; i < end; i++) {
    for (int j = 0; j < size; j++) (*A)[i*size + j] = PI * (float) i + j;
  }
  return 0;
}


int matrix_parallel_mult
(const int size, const int thread_number, const float* A, const float* B, float* result, const int start, const int end)
{
  if (thread_number > size) return ERROR_TOO_MANY_THREADS;
  if (thread_number < 2) return ERROR_TOO_FEW_THREADS;

  int i, j;
  #pragma omp for private(i,j) collapse(2) num_threads(thread_number)
    for (i = start; i < end; i++) {
      for (j = 0; j < size; j++) {
        float temp = (float) 0.;
        for (int k = 0; k < size; k++) temp += A[i*size + k] * B[j*size + k];
        result[i*size + j] = temp;
      }
    }


  return 0;
}

void matrix_print(const int rows, const int columns, const float* A) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < columns; j++) {
      printf("%f ", A[i*columns + j]);
    }
    printf("\n");
  }
}



int main(int argc, char** argv) {
  float *A, *B, *result;
  int start, end;

  MPI_Init(&argc, &argv);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == MASTER_NODE) {
    start = 0;
    end = MATRIX_SIZE / 2;

    matrix_init(MATRIX_SIZE, &A, start, end);
    matrix_init(MATRIX_SIZE, &B, 0, MATRIX_SIZE);
    MPI_Send(B, MATRIX_SIZE * MATRIX_SIZE, MPI_FLOAT, SLAVE_NODE, 0, MPI_COMM_WORLD);

  }
  else {
    start = MATRIX_SIZE / 2;
    end = MATRIX_SIZE;

    matrix_allocate(MATRIX_SIZE, &B);

    matrix_init(MATRIX_SIZE, &A, start, end);
    MPI_Recv(B, MATRIX_SIZE * MATRIX_SIZE, MPI_FLOAT, MASTER_NODE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }


  matrix_allocate(MATRIX_SIZE, &result);
  for (int i = start; i < end; i++) {
    for (int j = 0; j < MATRIX_SIZE; j++) result[i*MATRIX_SIZE + j] = (float) 0.;
  }

  matrix_parallel_mult(MATRIX_SIZE, THREAD_NUMBER, A, B, result, start, end);

  if (rank == MASTER_NODE) {
    MPI_Recv(result + (MATRIX_SIZE/ 2) * MATRIX_SIZE,
             (MATRIX_SIZE / 2) * MATRIX_SIZE,
             MPI_FLOAT,
             SLAVE_NODE,
             1,
             MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
  } else {
    MPI_Send(result + (MATRIX_SIZE / 2) * MATRIX_SIZE,
             (MATRIX_SIZE / 2) * MATRIX_SIZE,
             MPI_FLOAT,
             MASTER_NODE,
             1,
             MPI_COMM_WORLD);
  }


  free(B);
  free(A);
  free(result);

  MPI_Finalize();

  matrix_print(MATRIX_SIZE, MATRIX_SIZE, result);
  return 0;
}