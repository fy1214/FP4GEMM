#include "ops.h"
#include "cuda_utils.h"

#include <torch/library.h>
#include <torch/version.h>
#include <torch/extension.h>

// A version of the TORCH_LIBRARY macro that expands the NAME, i.e. so NAME
// could be a macro instead of a literal token.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("scaled_fp4_quant", &scaled_fp4_quant, "scaled fp4 quant act (CUDA)");
    m.def("scaled_fp4_dequant", &scaled_fp4_dequant, "scaled fp4 dequant act (CUDA)");
}
