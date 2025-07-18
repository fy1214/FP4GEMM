#include <cutlass/fast_math.h>
#include <torch/extension.h>

#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <type_traits>

#include "kernel_traits.h"

using namespace cute;

template <typename FlashAttnConfig_>
__global__ void fp8_gemm_cute_kernel(typename FlashAttnConfig_::Type *pA,
                                     typename FlashAttnConfig_::Type *pB,
                                     typename FlashAttnConfig_::OUT_TYPE *pD, 
                                     int m, int n, int k) {
  using namespace cute;
  using T = typename FlashAttnConfig_::Type;
  constexpr int BM = FlashAttnConfig_::kBlockM;
  constexpr int BN = FlashAttnConfig_::kBlockN;
  constexpr int BK = FlashAttnConfig_::kBlockK;
  constexpr int kStage = FlashAttnConfig_::kStage;

  using TiledCopy = typename FlashAttnConfig_::TiledCopyABC;
  using SmemCopyAtom = typename FlashAttnConfig_::SmemCopyAtom;
  using TiledMMA = typename FlashAttnConfig_::TiledMma;
  using SmemCopyAtomO = typename FlashAttnConfig_::SmemCopyAtomO;

  const int bx = blockIdx.x, by = blockIdx.y;
  const int tx = threadIdx.x;

  auto A =
      make_tensor(make_gmem_ptr(pA),
                  make_layout(make_shape(m, k), GenRowMajor{}));
  auto B =
      make_tensor(make_gmem_ptr(pB),
                  make_layout(make_shape(n, k), GenRowMajor{}));
  auto D =
      make_tensor(make_gmem_ptr(pD),
                  make_layout(make_shape(m, n), GenRowMajor{}));
  
  /**
  auto cta_tiler = make_shape(BM, BN, BK);
  auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _); // (m,n,k)
  auto gA = local_tile(A, cta_tiler, cta_coord, Step<_1, X,_1>{}); // (BLK_M,BLK_K,k)
  auto gB = local_tile(B, cta_tiler, cta_coord, Step< X,_1,_1>{}); // (BLK_N,BLK_K,k)
  auto gD = local_tile(D, cta_tiler, cta_coord, Step<_1,_1, X>{}); // (BLK_N,BLK_K,k)
  */
  
  Tensor gA = local_tile(A, make_tile(Int<BM>{}, Int<BK>{}),
                         make_coord(bx, _)); // (BM, BK, num_tile_k)
  Tensor gB = local_tile(B, make_tile(Int<BN>{}, Int<BK>{}),
                         make_coord(by, _)); // (BN, BK, num_tile_k)
  Tensor gD = local_tile(D, make_tile(Int<BM>{}, Int<BN>{}),
                         make_coord(bx, by)); // (BM, BN)

  __shared__ T psA[BM * BK * kStage], psB[BN * BK * kStage];

  auto sA = make_tensor(
      make_smem_ptr(psA),
      make_layout(make_shape(Int<BM>{}, Int<BK>{}, Int<kStage>{}), GenRowMajor{}));
  auto sB = make_tensor(
      make_smem_ptr(psB),
      make_layout(make_shape(Int<BN>{}, Int<BK>{}, Int<kStage>{}), GenRowMajor{}));
  
  // global memory -> shared memory
  TiledCopy tiled_copy;
  auto g2s_thr_copy = tiled_copy.get_slice(tx);
  auto tAgA = g2s_thr_copy.partition_S(gA);
  auto tAsA = g2s_thr_copy.partition_D(sA);
  auto tBgB = g2s_thr_copy.partition_S(gB);
  auto tBsB = g2s_thr_copy.partition_D(sB);

  // TiledMMA
  TiledMMA tiled_mma;
  auto thr_mma = tiled_mma.get_slice(tx);
  auto tCrA = thr_mma.partition_fragment_A(gA(_, _, 0)); // (MMA, MMA_M, MMA_K)
  auto tCrB = thr_mma.partition_fragment_B(gB(_, _, 0)); // (MMA, MMA_N, MMA_K)
  // auto tCrD = thr_mma.partition_fragment_C(gD);          // (MMA, MMA_M, MMA_N)
  auto tCrD = partition_fragment_C(tiled_mma, Shape<Int<BM>, Int<BN>>{});
  clear(tCrD);

  // shared memory -> register memory in TiledMMA
  auto tiled_s2r_copy_A = make_tiled_copy_A(SmemCopyAtom{}, tiled_mma);
  auto s2r_thr_copy_a = tiled_s2r_copy_A.get_slice(tx);
  auto tArA = s2r_thr_copy_a.partition_S(sA);     // (CPY, CPY_M, CPY_K, kStage)
  auto tCrA_view = s2r_thr_copy_a.retile_D(tCrA); // (CPY, CPY_M, CPY_K)

  auto tiled_s2r_copy_B = make_tiled_copy_B(SmemCopyAtom{}, tiled_mma);
  auto s2r_thr_copy_b = tiled_s2r_copy_B.get_slice(tx);
  auto tBrB = s2r_thr_copy_b.partition_S(sB);     // (CPY, CPY_N, CPY_K, kStage)
  auto tCrB_view = s2r_thr_copy_b.retile_D(tCrB); // (CPY, CPY_N, CPY_K)

  if (thread0()) {
    print("tiled_copy: "); print(tiled_copy); print("\n");
    print("tAgA: "); print(tAgA.layout()); print("\n");
    print("tAsA: "); print(tAsA.layout()); print("\n");
    print("tBgB: "); print(tBgB.layout()); print("\n");
    print("tBsB: "); print(tBsB.layout()); print("\n");
    
    print("tiled_mma: "); print(tiled_mma); print("\n");
    print("tCrA: "); print(tCrA.layout()); print("\n");
    print("tCrB: "); print(tCrB.layout()); print("\n");
    print("tCrD: "); print(tCrD.layout()); print("\n");

    print("tiled_s2r_copy_A: "); print(tiled_s2r_copy_A); print("\n");
    print("tAsA: "); print(tAsA.layout()); print("\n");
    print("tCrA_view: "); print(tCrA_view.layout()); print("\n");

    print("tiled_s2r_copy_B: "); print(tiled_s2r_copy_B); print("\n");
    print("tBsB: "); print(tBsB.layout()); print("\n");
    print("tCrB_view: "); print(tCrB_view.layout()); print("\n");
  }
  
  auto old_layout = tAsA.layout();
  auto new_layout = recast_layout<cutlass::float_e4m3_t,uint128_t>(old_layout);

  // PREFETCH
  // submit kStage - 1 tile
  // gmem -> shm
  int itile_to_read = 0;
  int ismem_read = 0;
  int ismem_write = 0;

  for (int istage = 0; istage < kStage - 1; ++istage) {
    cute::copy(g2s_thr_copy, tAgA(_, _, _, istage),
               tAsA(_, _, _, istage));
    cute::copy(g2s_thr_copy, tBgB(_, _, _, istage),
               tBsB(_, _, _, istage));
    cp_async_fence();

    ++itile_to_read;
    ++ismem_write;
  }

  // wait one submitted gmem->smem done
  cp_async_wait<kStage - 2>();
  __syncthreads();

  // smem -> reg
  // tAsA: (CPY, CPY_M, CPY_K, kStage) tCrA_view: (CPY, CPY_M, CPY_K)
  cute::copy(tiled_s2r_copy_A, tAsA(_, 0, _, ismem_read), tCrA_view(_, 0, _));
  cute::copy(tiled_s2r_copy_B, tBsB(_, 0, _, ismem_read), tCrB_view(_, 0, _));

  // loop over k: i. load tile, ii. mma
  int ntile = k / BK;
#pragma unroll 1
  for (int itile = 0; itile < ntile; ++itile) {
    int nk = size<1>(tCrA); // (MMA, MMA_M, MMA_K)
    
#pragma unroll
    for (int ik = 0; ik < nk; ++ik) {
      int ik_next = (ik + 1) % nk;
      if (ik == nk - 1) {
        cp_async_wait<kStage - 2>();
        __syncthreads();

        ismem_read = (ismem_read + 1) % kStage;
      }

      // shm -> reg s[itile][ik + 1] -> r[ik + 1]
      // tAsA: (CPY, CPY_M, CPY_K, kStage), tCrA_view: (CPY, CPY_M, CPY_K)
      cute::copy(tiled_s2r_copy_A, tAsA(_, ik_next, _, ismem_read),
                 tCrA_view(_, ik_next, _));
      // tBsB: (CPY, CPY_M, CPY_K, kStage), tCrB_view: (CPY, CPY_M, CPY_K)
      cute::copy(tiled_s2r_copy_B, tBsB(_, ik_next, _, ismem_read),
                 tCrB_view(_, ik_next, _));
      
      if (ik == 0) {
        if (itile_to_read < ntile) {
          cute::copy(g2s_thr_copy, tAgA(_, _, _, itile_to_read),
                     tAsA(_, _, _, ismem_write));
          cute::copy(g2s_thr_copy, tBgB(_, _, _, itile_to_read),
                     tBsB(_, _, _, ismem_write));
          ++itile_to_read;
          ismem_write = (ismem_write + 1) % kStage;
        }

        cp_async_fence();
      }

      cute::gemm(tiled_mma, tCrD, tCrA(_, ik, _), tCrB(_, ik, _), tCrD);
    }
  }
  // copy O back to gmem
  auto tiled_r2s_copy_O = make_tiled_copy_C(SmemCopyAtomO{}, tiled_mma);
  auto thr_r2s_copy_O = tiled_r2s_copy_O.get_slice(tx);
  auto tXrD = thr_r2s_copy_O.retile_S(tCrD);
  auto tXsD = thr_r2s_copy_O.partition_D(gD);
  cute::copy(tiled_r2s_copy_O, tXrD, tXsD);
}

void fp8_gemm_cute(torch::Tensor A, torch::Tensor B, torch::Tensor D) {
  
  using config = Flash_kernel_traits<128,128,128,4,2,
    cutlass::float_e4m3_t,cutlass::float_e4m3_t,float>;
  using T = config::Type;
  using O_T = config::OUT_TYPE;

  const int m = A.size(0); // m, n, k
  const int n = B.size(0);
  const int k = A.size(1);

  dim3 block(size(config::kNThreads));
  dim3 grid(ceil_div(m, config::kBlockM),
            ceil_div(n, config::kBlockN));
  fp8_gemm_cute_kernel<config><<<grid, block>>>(
      reinterpret_cast<T *>(A.data_ptr()),
      reinterpret_cast<T *>(B.data_ptr()),
      reinterpret_cast<O_T *>(D.data_ptr()),
      m, n, k);
}