#include <torch/all.h>

#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include "utils.cuh"
#include "../cuda_utils.h"

#define ELTS_PER_THREAD 8

constexpr int CVT_FP4_ELTS_PER_TIME = 8;
constexpr int CVT_FP4_SF_VEC_SIZE = 16;

__device__ __nv_bfloat16 half_to_bf16_fast(__half h) {
    unsigned short fp16_bits = *reinterpret_cast<unsigned short*>(&h);
    unsigned int fp32_bits = ((fp16_bits & 0x8000) << 16) |  // 符号位
                            ((fp16_bits & 0x7C00) << 13) |  // 指数位
                            ((fp16_bits & 0x03FF) << 13);   // 尾数位扩展
    unsigned short bf16_bits = fp32_bits >> 16;             // 截断为 BF16
    return *reinterpret_cast<__nv_bfloat16*>(&bf16_bits);
}

// Use UE4M3 by default.
template <class Type, bool UE8M0_SF = false>
__global__ void
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
__launch_bounds__(512, 4) cvt_fp4_to_fp16(
#else
cvt_fp4_to_fp16(
#endif
    int32_t numRows, int32_t numCols, uint32_t const* in, __nv_fp8_e4m3 const* SFScale,Type* out) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)

  // Input tensor row/col loops.
  for (int rowIdx = blockIdx.x; rowIdx < numRows; rowIdx += gridDim.x) {
    for (int colIdx = threadIdx.x; colIdx < numCols; colIdx += blockDim.x) {
      int64_t inOffset = rowIdx * numCols + colIdx;
      int64_t outOffset = rowIdx * numCols * CVT_FP4_ELTS_PER_TIME + colIdx * CVT_FP4_ELTS_PER_TIME;
      // the scale from fp8 to half
      __nv_fp8_e4m3 e4m3_scale_val = SFScale[inOffset];
      // __nv_fp8_storage_t e4m3_scale_storage = __nv_fp8_e4m3_to_storage(e4m3_scale_val);
      // __half_raw half_scale_raw = __nv_cvt_fp8_to_halfraw(e4m3_scale_storage, __NV_E4M3);
      half half_scale = static_cast<__half>(e4m3_scale_val); ;

      // the input from fp4 to half
      __nv_fp4_storage_t const* tmp = reinterpret_cast<__nv_fp4_storage_t const*>(in + inOffset);
      for (int i = 0; i < CVT_FP4_ELTS_PER_TIME; i++) {
        __nv_fp4_storage_t fp4_elet = tmp[i];
        __half_raw half_raw_elet = __nv_cvt_fp4_to_halfraw(fp4_elet, __NV_E2M1);
        half half_elet = static_cast<half>(half_raw_elet);
        half res = __hmul(half_scale, half_elet);
        if constexpr (std::is_same_v<Type, __nv_bfloat16>) {
            out[outOffset + i] = half_to_bf16_fast(res);
        }else {
            out[outOffset + i] = res;
        }
      }
    }
  }
#endif
}

template <typename T>
void invokeFP4deQuantization(int m, int n, uint32_t const* input, __nv_fp8_e4m3 const* SFScale,
                           T* output, bool useUE8M0, int multiProcessorCount, cudaStream_t stream) {
  // Grid, Block size.
  // Each thread converts 8 values.
  dim3 block(std::min(int(n), 512));
  // Get number of blocks per SM (assume we can fully utilize the SM).
  int const numBlocksPerSM = 2048 / block.x;
  dim3 grid(std::min(int(m), multiProcessorCount * numBlocksPerSM));

  // Launch the cvt kernel.
  if (useUE8M0) {
    cvt_fp4_to_fp16<T, true><<<grid, block, 0, stream>>>(
        m, n, input, SFScale, reinterpret_cast<T*>(output));
  } else {
    cvt_fp4_to_fp16<T, false><<<grid, block, 0, stream>>>(
        m, n, input, SFScale, reinterpret_cast<T*>(output));
  }
}

// Instantiate the function.
template void invokeFP4deQuantization(int m, int n, uint32_t const* input,
                                      __nv_fp8_e4m3 const* SFScale, half* output,
                                      bool useUE8M0, int multiProcessorCount,
                                      cudaStream_t stream);

template void invokeFP4deQuantization(int m, int n, uint32_t const* input,
                                      __nv_fp8_e4m3 const* SFScale, __nv_bfloat16* output,
                                      bool useUE8M0, int multiProcessorCount,
                                      cudaStream_t stream);

void scaled_fp4_dequant(torch::Tensor const& output,
                        torch::Tensor const& input,
                        torch::Tensor const& input_sf) {
  int32_t m = input.size(0);
  int32_t n = input.size(1);

  TORCH_CHECK(n % 16 == 0, "The N dimension must be multiple of 16.");

  int multiProcessorCount =
      get_device_attribute(cudaDevAttrMultiProcessorCount, -1);

  auto input_ptr = static_cast<uint32_t const*>(input.data_ptr());
  auto input_sf_ptr = static_cast<__nv_fp8_e4m3 const*>(input_sf.data_ptr());
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  auto stream = at::cuda::getCurrentCUDAStream(input.get_device());

  // We don't support e8m0 scales at this moment.
  bool useUE8M0 = false;

  switch (output.scalar_type()) {
    case torch::kHalf: {
      auto output_ptr = reinterpret_cast<half const*>(output.data_ptr());
      invokeFP4deQuantization(m, n, input_ptr, input_sf_ptr, output_ptr,
                            useUE8M0, multiProcessorCount, stream);
      break;
    }
    case torch::kBFloat16: {
      auto output_ptr = reinterpret_cast<__nv_bfloat16 const*>(output.data_ptr());
      invokeFP4deQuantization(m, n, input_ptr, input_sf_ptr, output_ptr,
                            useUE8M0, multiProcessorCount, stream);
      break;
    }
    default: {
      std::cerr << "Observing: " << input.scalar_type()
                << " for the input datatype which is invalid";
      throw std::runtime_error(
          "Unsupported input data type for quantize_to_fp4.");
    }
  }
}
