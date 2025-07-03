from mimetypes import add_type

import torch

import fp4_gemm

if __name__ == '__main__':
    m, n = 1024, 512

    x = torch.rand((m, n), device='cuda', dtype=torch.bfloat16)
    x_scale = torch.tensor([1], device='cuda', dtype=torch.float32)

    out = torch.empty((m, n // 8), device='cuda', dtype=torch.int32)
    out_scale = torch.empty((m, n // 8), device='cuda', dtype=torch.float8_e4m3fn)
    torch.ops.fp4_gemm.scaled_fp4_quant(out, x, out_scale, x_scale)
