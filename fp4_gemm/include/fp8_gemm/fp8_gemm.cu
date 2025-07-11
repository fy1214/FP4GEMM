#include <cutlass/fast_math.h>
#include <torch/extension.h>

#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <type_traits>

#include "kernel_traits.h"

using namespace cute;

template <typename FlashAttnConfig_>
__global__ void fp8_gemm_cute_kernel(typename FlashAttnConfig_::T *pA,
                                     typename FlashAttnConfig_::T *pB,
                                     typename FlashAttnConfig_::T *pC, 
                                     int m, int n, int k) {
  using namespace cute;
  using T = typename FlashAttnConfig_::Type;
  constexpr int kBlockM = FlashAttnConfig_::kBlockM;
  constexpr int kBlockN = FlashAttnConfig_::kBlockN;
  constexpr int kBlockK = FlashAttnConfig_::kBlockK;
  using TiledCopy = typename FlashAttnConfig_::TiledCopyABC;

  const int bx = blockIdx.x, by = blockIdx.y;
  const int tx = threadIdx.x;

  auto A =
      make_tensor(make_gmem_ptr(pA),
                  make_layout(make_shape(m, k), GenRowMajor{}));
  auto B =
      make_tensor(make_gmem_ptr(pB),
                  make_layout(make_shape(n, k), GenRowMajor{}));
  auto C =
      make_tensor(make_gmem_ptr(pC),
                  make_layout(make_shape(m, n), GenRowMajor{}));
  
  auto cta_tiler = make_shape(kBlockM, kBlockN, kBlockK);
  auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _); // (m,n,k)
  auto gA = local_tile(A, cta_tiler, cta_coord, Step<_1, X,_1>{}); // (BLK_M,BLK_K,k)
  auto gB = local_tile(B, cta_tiler, cta_coord, Step< X,_1,_1>{}); // (BLK_N,BLK_K,k)
  auto gC = local_tile(C, cta_tiler, cta_coord, Step<_1,_1, X>{}); // (BLK_N,BLK_K,k)
  
  __shared__ T psA[kBlockM * kBlockK], psB[kBlockN * kBlockK];

  auto sA = make_tensor(
      make_smem_ptr(psA),
      make_layout(make_shape(Int<kBlockM>{}, Int<kBlockK>{}), GenRowMajor{}));
  auto sB = make_tensor(
      make_smem_ptr(psB),
      make_layout(make_shape(Int<kBlockN>{}, Int<kBlockK>{}), GenRowMajor{}));
  
  TiledCopy tiled_copy;
  auto thr_copy = tiled_copy.get_slice(tx);
  auto tAgA = thr_copy.partition_S(gA);
  auto tAsA = thr_copy.partition_D(sA);
  auto tBgB = thr_copy.partition_S(gB);
  auto tBsB = thr_copy.partition_D(sB);

  // copy A into smem
  copy(tiled_copy, tAgA, tAsA);

  if (thread0()) {
    print("tAgA: "); print(tAgA.layout()); print("\n");
    print("tAsA: "); print(tAsA.layout()); print("\n");
    print("tBgB: "); print(tBgB.layout()); print("\n");
    print("tBsB: "); print(tBsB.layout()); print("\n");
    for (int i = 0; i < size(tQsQ); i++) {
        printf("tBsB(%d): ", i); print(tBsB(iu)); print("\n");
    }
  }
}

void fp8_gemm_cute(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
  
  using config = Flash_kernel_traits;
  using T = config::Type;

  assert(sanity_check(Q, K, V, O));
  const int m = A.size(0); // m, n, k
  const int n = B.size(1);
  const int k = A.size(2);

  dim3 block(size(config::kNThreads));
  dim3 grid(size(ceil_div(m, config::kBlockM),
                 ceil_div(n, config::kBlockN)));
  fp8_gemm_cute_kernel<config><<<grid, block>>>(
      reinterpret_cast<T *>(A.data_ptr()),
      reinterpret_cast<T *>(B.data_ptr()),
      reinterpret_cast<T *>(C.data_ptr()),
      m, n, k);
}