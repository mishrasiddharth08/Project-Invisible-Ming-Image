"""Capability-based device detection and VRAM-tier policy. Installs nothing."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Hardware:
    device: str
    backend: str            # 'cuda' | 'rocm' | 'cpu'
    bf16: bool
    total_gib: float


def detect():
    import torch
    if torch.cuda.is_available():
        index = torch.cuda.current_device()
        backend = 'rocm' if getattr(torch.version, 'hip', None) else 'cuda'
        try:
            bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
        except TypeError:
            bf16 = torch.cuda.is_bf16_supported()
        total = torch.cuda.get_device_properties(index).total_memory / 2**30
        return Hardware(f'cuda:{index}', backend, bool(bf16), total)
    return Hardware('cpu', 'cpu', True, 0.0)


def is_packed(path):
    """True when a safetensors file uses packed/quantized formats (int8_convrot, w4a8, fp8/fp4)."""
    from .assets import header
    data = header(path)
    metadata = data.get('__metadata__', {})
    if metadata.get('_quantization_metadata'):
        return True
    return any(name.endswith(('.weight_scale', '.comfy_quant', '.weight_scale_2'))
               or (isinstance(value, dict) and value.get('dtype', '').startswith(('F8', 'F4', 'I4', 'U4', 'I8')))
               for name, value in data.items())


def tier(hw):
    """VRAM tier: what the 6B DiT + Ling TE + VAE can hold per pass."""
    gib = hw.total_gib
    if hw.backend == 'cpu':
        return 'cpu'
    if gib >= 32:
        return '32'
    if gib >= 20:
        return '24'
    if gib >= 16:
        return '16'
    if gib >= 12:
        return '12'
    if gib >= 8:
        return '8'
    return '6'


# Preferred formats per component per tier, most preferred first.
# NVIDIA: int8_convrot and w4a8 need CUDA kernels; ROCm/CPU use BF16 only.
PREFERENCE = {
    'cuda': {
        '32': {'dit': ['int8_convrot', 'bf16'], 'clip': ['w4a8', 'int8_convrot', 'bf16']},
        '24': {'dit': ['int8_convrot', 'bf16'], 'clip': ['w4a8', 'int8_convrot', 'bf16']},
        '16': {'dit': ['int8_convrot', 'bf16'], 'clip': ['w4a8', 'int8_convrot', 'bf16']},
        '12': {'dit': ['int8_convrot', 'bf16'], 'clip': ['w4a8', 'int8_convrot', 'bf16']},
        '8':  {'dit': ['int8_convrot', 'bf16'], 'clip': ['w4a8', 'int8_convrot', 'bf16']},
        '6':  {'dit': ['int8_convrot', 'bf16'], 'clip': ['w4a8', 'int8_convrot', 'bf16']},
    },
    'rocm': {
        '*': {'dit': ['bf16'], 'clip': ['bf16']},
    },
    'cpu': {
        '*': {'dit': ['bf16'], 'clip': ['bf16']},
    },
}


def preferences(hw, role):
    table = PREFERENCE.get(hw.backend) or PREFERENCE['cuda']
    return table.get(tier(hw), table.get('*', {})).get(role, [])