#include "cuda_utils.h"
#include "ops.h"

#include <torch/library.h>
#include <torch/version.h>

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("scale the input fp16/bf16 tensor into fp4");
  ops.impl("scaled_fp4_quant", torch::kCUDA, &scaled_fp4_quant);
}
