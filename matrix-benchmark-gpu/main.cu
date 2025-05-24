#include <stdio.h>

__global__ void add(const int *a, const int *b, int *c) {
  *c += *a + *b;
}

int main() {
  int a, b, c;
  int *a_ptr, *b_ptr, *c_ptr;
  const int size = sizeof(int);

  cudaMalloc((void **) &a_ptr, size);
  cudaMalloc((void **) &b_ptr, size);
  cudaMalloc((void **) &c_ptr, size);

  a = 2;
  b = 3;
  c = 5;

  cudaMemcpy(a_ptr, &a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(b_ptr, &b, size, cudaMemcpyHostToDevice);
  cudaMemcpy(c_ptr, &c, size, cudaMemcpyHostToDevice);

  add<<<1,1>>>(a_ptr, b_ptr, c_ptr);

  cudaMemcpy(&c, c_ptr, size, cudaMemcpyDeviceToHost);

  printf("The sum is: %d\n", c);
  return 0;
}
