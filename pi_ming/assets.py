"""Read tensor headers before accepting a component; never load unknown pickle files."""
import json
import struct
from pathlib import Path
from .config import roots


def header(path):
    path = Path(path)
    if path.suffix.lower() != '.safetensors':
        raise ValueError('Ming-Image accepts .safetensors model files only.')
    with path.open('rb') as stream:
        size = stream.read(8)
        if len(size) != 8:
            raise ValueError('Incomplete model: ' + str(path))
        n = struct.unpack('<Q', size)[0]
        if not 2 <= n <= 64 * 1024 * 1024:
            raise ValueError('Invalid model header: ' + str(path))
        raw = stream.read(n)
        if len(raw) != n:
            raise ValueError('Incomplete model header: ' + str(path))
    try:
        data = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError('Invalid model header: ' + str(path)) from error
    if not isinstance(data, dict):
        raise ValueError('Invalid model header: ' + str(path))
    tensors = {k: v for k, v in data.items() if k != '__metadata__'}
    if not tensors:
        raise ValueError('Empty model: ' + str(path))
    spans = []
    for name, value in tensors.items():
        if not isinstance(name, str) or not isinstance(value, dict):
            raise ValueError('Invalid tensor header: ' + str(path))
        offsets = value.get('data_offsets')
        shape = value.get('shape')
        if (not isinstance(offsets, list) or len(offsets) != 2 or
                any(type(x) is not int for x in offsets) or offsets[0] < 0 or offsets[1] < offsets[0] or
                not isinstance(shape, list) or any(type(x) is not int or x < 0 for x in shape)):
            raise ValueError('Invalid tensor header: ' + str(path))
        dtype = value.get('dtype')
        if dtype not in {'BOOL', 'U8', 'I8', 'I16', 'I32', 'I64', 'F16', 'BF16', 'F32', 'F64',
                         'F8_E4M3', 'F8_E5M2', 'F4_E2M1', 'U4', 'I4'}:
            raise ValueError('Unsupported tensor dtype ' + repr(dtype) + ': ' + str(path))
        spans.append(tuple(offsets))
    cursor = 0
    for start, end in sorted(spans):
        if start != cursor:
            raise ValueError('Invalid tensor offsets: ' + str(path))
        cursor = end
    if path.stat().st_size < 8 + n + cursor:
        raise ValueError('Download is incomplete: ' + str(path))
    return tensors


def classify(path):
    name = Path(path).name.lower().replace('-', '_')
    if 'ming' not in name:
        return None
    if 'vae' in name:
        return 'vae'
    if 'ling_mini' in name:
        return 'clip'
    if 'lora' in name or 'adapter' in name or 'turbo' in name:
        return 'lora'
    if 'ming_image' in name and 'layer' in name:
        return 'dit_layer'
    if 'ming_image' in name:
        return 'dit'
    return None


def scan(search_roots=None):
    found = {k: [] for k in ('dit', 'dit_layer', 'clip', 'clip_layer', 'vae', 'lora')}
    seen = set()
    for root in search_roots if search_roots is not None else roots():
        root = Path(root)
        if not root.is_dir():
            continue
        for path in root.rglob('*.safetensors'):
            value = str(path.resolve())
            role = classify(path)
            if role == 'dit_layer':
                # Layer variant doubles as its own checkpoint and supplies its text encoder name.
                if value not in seen:
                    seen.add(value)
                    found['dit_layer'].append(value)
            elif role and value not in seen:
                seen.add(value)
                found[role].append(value)
    for path in found['dit_layer']:
        name = Path(path).name.lower()
        if 'layer' in name:
            for clip in found['clip']:
                if 'layer' in Path(clip).name.lower():
                    found['clip_layer'].append(clip)
    return {k: sorted(v) for k, v in found.items()}


def resolve(selection, role, inventory, hw=None):
    if selection and selection not in ('Auto', '(none)'):
        path = str(Path(selection).resolve())
        if path not in inventory.get(role, []):
            raise ValueError(f'Selected {role} is missing or outside the Ming-Image model folders: {path}')
    else:
        choices = inventory.get(role) or []
        if not choices:
            raise ValueError(f'Ming-Image {role} is missing. Open Ming controls → Models for download links and folder instructions.')
        if role in ('dit', 'dit_layer', 'clip', 'clip_layer'):
            from . import hardware
            if hw is None:
                hw = hardware.detect()
            # Explicit user selection always wins; otherwise follow the VRAM-tier
            # format preference and skip packed formats the backend cannot run.
            preferred = []
            for fmt in hardware.preferences(hw, 'dit' if role.startswith('dit') else 'clip'):
                preferred.extend(p for p in choices if fmt in p.lower())
            choices = preferred or choices
        path = choices[0]
    tensors = header(path)
    required = {
        'dit': ('img_patch_embed', 'final_layer.linear.weight'),
        'dit_layer': ('img_patch_embed', 'final_layer.linear.weight'),
        'clip': ('thinker.norm.weight', 'thinker.layers.1.mlp.image_gate.proj.weight'),
        'clip_layer': ('thinker.norm.weight', 'thinker.layers.1.mlp.image_gate.proj.weight'),
        'vae': ('decoder.up_blocks.0.resnets.0.conv1.weight', 'encoder.down_blocks.0.resnets.0.conv1.weight'),
    }
    if not all(any(key.endswith(suffix) or suffix in key for key in tensors) for suffix in required.get(role, ())):
        raise ValueError(f'Incompatible Ming-Image {role} tensor structure: {path}')
    return path


def fingerprint(paths):
    return tuple((str(p), Path(p).stat().st_size, Path(p).stat().st_mtime_ns) for p in paths)