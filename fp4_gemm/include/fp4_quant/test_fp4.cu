#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include <cuda_fp8.h>
#include <cuda_fp4.h>

__inline__ __device__ __nv_fp4_e2m1 cvt_bf16_to_nv_fp4_e2m1(const __nv_bfloat16 in) {
  __nv_fp4_e2m1 tmp = __nv_fp4_e2m1(in);
  return tmp;
}

__global__  void test_fun(__nv_bfloat16* a, __nv_fp4_e2m1* b, int n) {
  int tid = threadIdx.x;
  for (int i = tid; i < n; i += blockIdx.x) {
    b[i] = cvt_bf16_to_nv_fp4_e2m1(a[i]);
  }
}

int main() {
  int n = 128;
  __nv_bfloat16* a;
  size_t size_a = n * sizeof(__nv_bfloat16);
  for (int i = 0; i < 2; i++) {
      a[i] = i;
  }
  a = (__nv_bfloat16*)malloc(size_a);

  __nv_fp4_e2m1* b;
  size_t size_b = n * sizeof(__nv_fp4_e2m1);
  b = (__nv_fp4_e2m1*)malloc(size_b);

  __nv_bfloat16* da;
  __nv_fp4_e2m1* db;
  cudaError_t err = cudaMalloc(&da, size_a);
  err = cudaMalloc(&db, size_b);
  cudaMemcpy(da, a, size_a, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    printf("CUDA Error: %s\n", cudaGetErrorString(err));
    return -1;
  }

  dim3 grid(1);
  dim3 block(128);

  test_fun<<<grid, block>>>(da, db, n);

  cudaMemcpy(b, db, size_b, cudaMemcpyDeviceToHost);

  for (int i = 0; i < 2; i++) {
    printf("%d \n", b[i]);
  }

  // 使用完毕后释放内存
  cudaFree(da);
  return 0;
}