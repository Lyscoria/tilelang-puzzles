# FlashAttention Extra

Minimal harness aligned with the `@T.prim_func` style:

- tensors use layout `[batch, seq_len, heads, dim]`
- kernels take explicit `Output` as the last argument
- `flash_attn` and `flash_attn2` are called as TileLang JIT factories

Run:

```bash
python3 extra/flash_attention_harness.py
```
