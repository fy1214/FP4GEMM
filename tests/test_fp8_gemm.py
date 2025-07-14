from typing import List, Tuple

import torch
import fp4_gemm

def per_token_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    pad_size = (128 - (n % 128)) % 128
    x = torch.nn.functional.pad(x, (0, pad_size), value=0) if pad_size > 0 else x
    x_view = x.view(m, -1, 128)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    fp8_data = (x_view * (448.0 / x_amax.unsqueeze(2))).to(torch.float8_e4m3fn)
    return fp8_data.view(m, n + pad_size)[:, :n], (x_amax / 448.0).view(m, -1)

if __name__ == '__main__':
    m, n, k = 1024, 1024, 2048

    a = torch.rand((m, k), device='cuda', dtype=torch.half)
    a_t, _ = per_token_cast_to_fp8(a)
    b = torch.rand((n, k), device='cuda', dtype=torch.half)
    b_t, _ = per_token_cast_to_fp8(b)
    c = torch.empty((m, n), device='cuda', dtype=torch.float8_e4m3fn)

    fp4_gemm.fp8_gemm_cute(a_t, b_t, c)
