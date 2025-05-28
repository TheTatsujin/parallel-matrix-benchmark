#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <papi.h>

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
(const int half_size, const int rank, const int thread_number, const float* A, const float* B, float* result)
{
  if (thread_number > half_size) return ERROR_TOO_MANY_THREADS;
  if (thread_number < 2) return ERROR_TOO_FEW_THREADS;

  const int retval = PAPI_thread_init((unsigned long (*)(void))omp_get_thread_num);
  if (retval != PAPI_OK) {
    fprintf(stderr, "PAPI thread initialization error. %d\n", retval);
    return -1;
  }

  int my_thread_id;
  #pragma omp parallel default(none) shared(A, B, result) firstprivate(half_size, rank) private(my_thread_id)
  {
    int EventSet = PAPI_NULL;
    long long measures[3];

    PAPI_register_thread();

    PAPI_create_eventset(&EventSet);
    PAPI_add_event(EventSet, PAPI_TOT_CYC);
    PAPI_add_event(EventSet, PAPI_SP_OPS);
    PAPI_add_event(EventSet, PAPI_L1_TCM);

    my_thread_id = (int) PAPI_thread_id();

    PAPI_start(EventSet);

    int i, j;
    #pragma omp for private(i,j) collapse(2)
    for (i = 0; i < half_size; i++) {
      for (j = 0; j < MATRIX_SIZE; j++) {
        float temp = (float) 0.;
        for (int k = 0; k < MATRIX_SIZE; k++) temp += A[i*MATRIX_SIZE + k] * B[j*MATRIX_SIZE + k];
        result[i*MATRIX_SIZE + j] = temp;
      }
    }
    PAPI_stop(EventSet, measures);

    #pragma omp critical
    {
      printf("Rank %d, Thread %d - Total Cycles: %lld\n", rank, my_thread_id, measures[0]);
      printf("Rank %d, Thread %d - float Precision Operations: %lld\n", rank, my_thread_id, measures[1]);
      printf("Rank %d, Thread %d - Total L1 Cache misses: %lld\n", rank, my_thread_id, measures[2]);
    }

    PAPI_cleanup_eventset(EventSet);
    PAPI_destroy_eventset(&EventSet);
    PAPI_unregister_thread();
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
  double start_time = 0.;
  float *A, *B, *result;
  float *full_result;
  const int master_matrix_size = MATRIX_SIZE / 2;
  const int slave_matrix_size = MATRIX_SIZE - master_matrix_size;
  int my_half_size;

  omp_set_num_threads(THREAD_NUMBER);

  const int retval = PAPI_library_init(PAPI_VER_CURRENT);
  if (retval != PAPI_VER_CURRENT) {
    fprintf(stderr, "PAPI init error: %d\n", retval);
    return -1;
  }

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

  if (rank == MASTER_NODE) start_time = MPI_Wtime();  // START TIME
  matrix_parallel_mult_half(my_half_size, rank, THREAD_NUMBER, A, B, result);

  if (rank == MASTER_NODE) {
    const double end_time = MPI_Wtime();    // STOP TIME
    const double elapsed_ms = (end_time - start_time) * 1000.0;  // seconds to ms

    printf("Elapsed time: %.5f ms\n", elapsed_ms);

    matrix_allocate(MATRIX_SIZE, &full_result);
    MPI_Gather(result, my_half_size * MATRIX_SIZE, MPI_FLOAT, full_result, slave_matrix_size * MATRIX_SIZE, MPI_FLOAT, MASTER_NODE, MPI_COMM_WORLD);
    printf("Press Enter for result\n");
    fgetc(stdin);
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