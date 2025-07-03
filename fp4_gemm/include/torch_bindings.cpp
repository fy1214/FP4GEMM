#include "ops.h"
#include "cuda_utils.h"

#include <torch/library.h>
#include <torch/version.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_bias_act", &fused_bias_act, "fused bias act (CUDA)");
}
