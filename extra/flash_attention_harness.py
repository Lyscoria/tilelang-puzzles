"""Tiny correctness and timing harness for FlashAttention kernels."""

import math

import torch
from flash_attention_impl import flash_attn, flash_attn2

batch = 1
heads = 8
seq_len = 512
dim = 64
block_M = 64
block_N = 64
thread_num = 128
is_causal = False

warmup = 10
repeat = 100
atol = 2e-2
rtol = 2e-2


def ref_attention(q, k, v):
    scores = torch.einsum("bqhd,bkhd->bhqk", q.float(), k.float()) / math.sqrt(dim)
    if is_causal:
        mask = torch.ones((seq_len, seq_len), dtype=torch.bool, device=q.device).triu(1)
        scores = scores.masked_fill(mask[None, None, :, :], float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("bhqk,bkhd->bqhd", probs, v.float()).to(q.dtype)


def bench_ms(fn):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeat


def run_one(name, kernel_func, q, k, v, expected):
    kernel = kernel_func(batch, heads, seq_len, dim, block_M, block_N, thread_num, is_causal)
    output = torch.zeros_like(q) if name == "flash_attn" else torch.empty_like(q)

    if name == "flash_attn":
        l_state = torch.zeros((batch, heads, seq_len), device="cuda", dtype=torch.float32)
        m_state = torch.full((batch, heads, seq_len), -float("inf"), device="cuda")
        kernel(q, k, v, output, l_state, m_state)
    else:
        kernel(q, k, v, output)

    ok = torch.allclose(output, expected, atol=atol, rtol=rtol)
    max_err = (output.float() - expected.float()).abs().max().item()
    mean_err = (output.float() - expected.float()).abs().mean().item()

    if name == "flash_attn":
        output.zero_()
        l_state.zero_()
        m_state.fill_(-float("inf"))

        def run_v1():
            output.zero_()
            l_state.zero_()
            m_state.fill_(-float("inf"))
            kernel(q, k, v, output, l_state, m_state)

        kernel_ms = bench_ms(run_v1)
    else:
        kernel_ms = bench_ms(lambda: kernel(q, k, v, output))

    print(f"{name}: ok={ok} max_err={max_err:.6g} mean_err={mean_err:.6g} time={kernel_ms:.4f} ms")
    return ok


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("This harness needs CUDA to run TileLang kernels.")

    torch.manual_seed(0)
    shape = (batch, seq_len, heads, dim)
    q = torch.randn(shape, device="cuda", dtype=torch.float16) * 0.5
    k = torch.randn(shape, device="cuda", dtype=torch.float16) * 0.5
    v = torch.randn(shape, device="cuda", dtype=torch.float16)

    expected = ref_attention(q, k, v)
    torch_ms = bench_ms(lambda: ref_attention(q, k, v))
    print(f"torch reference: {torch_ms:.4f} ms")

    ok1 = run_one("flash_attn", flash_attn, q, k, v, expected)
    ok2 = run_one("flash_attn2", flash_attn2, q, k, v, expected)
    raise SystemExit(0 if ok1 and ok2 else 1)


if __name__ == "__main__":
    main()
