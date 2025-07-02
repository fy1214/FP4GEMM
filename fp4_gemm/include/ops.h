#pragma once

#include <optional>
#include <torch/library.h>

void scaled_fp4_quant(
    torch::Tensor const& output,
    torch::Tensor const& input,
    torch::Tensor const& output_sf,
    torch::Tensor const& input_sf
    );