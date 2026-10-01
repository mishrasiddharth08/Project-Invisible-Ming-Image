# Quantization support

Formats are detected from the **tensor headers and live modules, never the filename**
(prompt rule 12/16/17). A descriptor can lie; the weights cannot.

## Component formats and where they run

| Component | Format | NVIDIA CUDA | AMD ROCm | CPU | Notes |
| --- | --- | --- | --- | --- | --- |
| DiT (`ming_image_0.1_design_*`) | `int8_convrot` | tested path | not verified | not verified | convrot rotates weights before quantising; loading goes through Forge's own quant operations |
| DiT | `bf16` | yes | yes | slow but correct | ~12.3 GB on disk |
| DiT Layer variant (`*_layer_*`) | `int8_convrot` / `bf16` | same as above | same | same | design-layer checkpoints; resolved as role `dit_layer` |
| Text encoder (`ming_image_0.1_ling_mini_2.0_*`) | `w4a8` | yes (CUDA kernels) | not verified | no | ~12.8 GB; smallest TE |
| Text encoder | `int8_convrot` | tested path | not verified | not verified | ~19.5 GB |
| Text encoder | `bf16` | yes | yes | slow | ~36.7 GB on disk |
| VAE | `bf16` | yes | yes | yes | ~254 MB |

## Automatic selection

When the user has **not** explicitly selected a component, `pi_ming/assets.py`
resolves by role following `pi_ming/hardware.py` preferences:

- **32/24/16/12/8/6 GB NVIDIA:** `int8_convrot` DiT, `w4a8` → `int8_convrot` → `bf16` TE.
- **ROCm / CPU:** `bf16` only. Packed convrot/w4a8 kernels are not verified
  outside CUDA and are never silently chosen there.

**The user's selection is an instruction, not a hint** (rule 36): an explicit
dropdown choice always wins over the automatic preference, and every file is
validated by tensor structure before load (rule 37), with one clear error line.

## LoRAs on quantised trunks

- **Never merged** into quantised weights (rule 13): convrot's rotated storage
  makes `weight * scale` an invalid dequantisation; a merged delta measured
  cosine ~0.57 in the reference build. Adapters are applied as a removable
  runtime patch on a disposable model clone and unpatched after each request
  (rules 14/15).
- Adapter keys that do not match the Ming model are rejected, never partially applied.

## Honest limits

- Only NVIDIA CUDA is tested. ROCm and CPU paths choose BF16 weights and are
  expected to work but are **not verified** — see VALIDATION.md once real
  hardware runs exist.
- `is_packed()` reads `__metadata__._quantization_metadata`, scale-prefixed
  tensor names (`*.weight_scale`, `*.comfy_quant`, `*.weight_scale_2`), and
  F8/F4/I4/U4/I8 dtypes. This covers int8_convrot, w4a8, nvfp4, mxfp8 and the
  fp8 family; new formats must be added to the check, not assumed covered.