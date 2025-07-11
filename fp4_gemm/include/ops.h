#pragma once

#include <optional>
#include <torch/library.h>
#include <torch/all.h>

void scaled_fp4_quant(
    torch::Tensor const& output,
    torch::Tensor const& input,
    torch::Tensor const& output_sf
    );

void scaled_fp4_dequant(torch::Tensor const& output,
                        torch::Tensor const& input,
                        torch::Tensor const& input_sf);

void fp8_gemm_cute(torch::Tensor A, torch::Tensor B, torch::Tensor C);
