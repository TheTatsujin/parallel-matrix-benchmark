#include <math.h>
#include <omp.h>
#include <papi.h>
#include <stdio.h>
#include <stdlib.h>

#define ERROR_TOO_FEW_THREADS 0x12;
#define ERROR_TOO_MANY_THREADS 0x54
#define ERROR_INCORRECT_THREAD_ID 0x65
#define ERROR_MEMORY_ALLOCATION 0x66

#define THREAD_NUMBER 6
#define MATRIX_SIZE 1200

#define PI 3.14159265358979323846


void matrix_allocate(const int size, float*** A) {
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

void matrix_free(const int size, float*** A, float*** B, float*** result) {
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

int matrix_mult
(const int size, float** A, float** B, float** result)
{
  int EventSet = PAPI_NULL;
  long long measures[3];
  PAPI_create_eventset(&EventSet);
  PAPI_add_event(EventSet, PAPI_TOT_CYC);
  PAPI_add_event(EventSet, PAPI_SP_OPS);
  PAPI_add_event(EventSet, PAPI_L1_TCM);

  matrix_transpose(size, B);

  PAPI_start(EventSet);
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      /// Row x Column
      result[i][j] = (float) 0.;
      for (int k = 0; k < size; k++) result[i][j] += A[i][k] * B[j][k];
    }
  }
  PAPI_stop(EventSet, measures);

  printf("Sequential - Total Cycles: %lld\n", measures[0]);
  printf("Sequential - float Precision Operations: %lld\n",measures[1]);
  printf("Sequential - Total L1 Cache misses: %lld\n",measures[2]);


  PAPI_cleanup_eventset(EventSet);
  PAPI_destroy_eventset(&EventSet);


  return 0;
}

void matrix_print(const int rows, const int columns, float** A) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < columns; j++) {
      printf("%f ", A[i][j]);
    }
    printf("\n");
  }
}

int matrix_parallel_mult
(const int size, const int thread_number, float** A, float** B, float** result)
{
  if (thread_number > size) return ERROR_TOO_MANY_THREADS;
  if (thread_number < 2) return ERROR_TOO_FEW_THREADS;

  const int retval = PAPI_thread_init((unsigned long (*)(void))omp_get_thread_num);
  if (retval != PAPI_OK) {
    fprintf(stderr, "PAPI thread initialization error. %d\n", retval);
    return -1;
  }

  matrix_transpose(size, B);

  /// Variable privada de cada hilo
  int my_thread_id;
  #pragma omp parallel default(none) shared(A, B, result) firstprivate(size) private(my_thread_id)
  {
    int i, EventSet = PAPI_NULL;
    long long measures[3];

    PAPI_create_eventset(&EventSet);
    PAPI_add_event(EventSet, PAPI_TOT_CYC);
    PAPI_add_event(EventSet, PAPI_SP_OPS);
    PAPI_add_event(EventSet, PAPI_L1_TCM);

    my_thread_id = (int) PAPI_thread_id();

    PAPI_start(EventSet);

    #pragma omp for private(i)
    for (i = 0; i < size; i++) {
      for (int j = 0; j < size; j++) {
        float temp = (float) 0.;
        for (int k = 0; k < size; k++) temp += A[i][k] * B[j][k];
        result[i][j] = temp;
      }
    }


    PAPI_stop(EventSet, measures);

    #pragma omp critical
    {
      printf("Thread %d - Total Cycles: %lld\n", my_thread_id, measures[0]);
      printf("Thread %d - float Precision Operations: %lld\n", my_thread_id, measures[1]);
      printf("Thread %d - Total L1 Cache misses: %lld\n", my_thread_id, measures[2]);
    }

    PAPI_cleanup_eventset(EventSet);
    PAPI_destroy_eventset(&EventSet);
  }

  return 0;
}


int main() {
  // Ideally 6 Threads for 6 Cores
  omp_set_num_threads(THREAD_NUMBER);

  // Multiplication Variables
  float** A;
  float** B;
  float **result;

  if (matrix_init(MATRIX_SIZE, &A, &B, &result) == ERROR_MEMORY_ALLOCATION) {
    fprintf(stderr, "Memory allocation error\n");
    matrix_free(MATRIX_SIZE, &A, &B, &result);
    return -1;
  }

  const int retval = PAPI_library_init(PAPI_VER_CURRENT);
  if (retval != PAPI_VER_CURRENT) {
    fprintf(stderr, "PAPI init error: %d\n", retval);
    matrix_free(MATRIX_SIZE, &A, &B, &result);
    return -1;
  }

  matrix_parallel_mult(MATRIX_SIZE, THREAD_NUMBER, A, B, result);
  matrix_mult(MATRIX_SIZE, A, B, result);

  printf("Press Enter for result\n");
  fgetc(stdin);
  matrix_print(MATRIX_SIZE, MATRIX_SIZE, result);

  matrix_free(MATRIX_SIZE, &A, &B, &result);

  return 0;
}
