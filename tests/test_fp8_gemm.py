from mimetypes import add_type

import torch

import fp4_gemm

if __name__ == '__main__':
    m, n, k = 1024, 1024, 2048

    a = torch.rand((m, k), device='cuda', dtype=torch.float8_e4m3fn)
    b = torch.rand((n, k), device='cuda', dtype=torch.float8_e4m3fn)
    c = torch.empty((m, n), device='cuda', dtype=torch.float8_e4m3fn)

    fp4_gemm.fp8_gemm_cute(a, b, c)
