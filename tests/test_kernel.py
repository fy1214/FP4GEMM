from mimetypes import add_type

import torch

import fp4_gemm

if __name__ == '__main__':
    m, n = 128, 128

    x = torch.rand((m, n), device='cuda', dtype=torch.half)

    out = torch.empty((m, n // 8), device='cuda', dtype=torch.int32)
    out_scale = torch.empty((m, n // 8), device='cuda', dtype=torch.float8_e4m3fn)
    fp4_gemm.scaled_fp4_quant(out, x, out_scale)

    print(out)
    print(out_scale)

    x_n = torch.empty_like(x)
    fp4_gemm.scaled_fp4_dequant(out, out_scale, x_n)
