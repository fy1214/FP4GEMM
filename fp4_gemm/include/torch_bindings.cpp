#include "ops.h"
#include "cuda_utils.h"

#include <torch/library.h>
#include <torch/version.h>

// A version of the TORCH_LIBRARY macro that expands the NAME, i.e. so NAME
// could be a macro instead of a literal token.
#define TORCH_LIBRARY_EXPAND(NAME, MODULE) TORCH_LIBRARY(NAME, MODULE)

TORCH_LIBRARY_EXPAND(fp4_gemm, ops) {
  ops.def("scaled_fp4_quant", &scaled_fp4_quant);
}
