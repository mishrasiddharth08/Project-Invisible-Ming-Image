"""Request-scoped Spectrum acceleration for the vendored ComfyUI backend.

The forecasting method is adapted from ComfyUI-Spectrum-QwenImage21 (MIT) via the
Project-Invisible Qwen-Image-2.1 port: a ridge Chebyshev prediction of the
pre-final-layer feature, blended with linear extrapolation. Here it hooks the
Lumina2-family FinalLayer of the Ming-Image DiT instead of a Diffusers pipeline.

Conservative by design: forecasts are skipped during warmup and the final steps,
never adjacent, and only when the predicted change is small relative to the
feature magnitude. At CFG 1 there is no second pass to skip, so it stays off.
"""
from __future__ import annotations

import contextlib
import types


@contextlib.contextmanager
def accelerate(model, enabled=False, steps=0, cfg=1.0, stop=None, stats=None):
    """Temporarily accelerate one request on a ComfyUI ModelPatcher's BaseModel.

    model: the ModelPatcher passed to comfy.sample.sample. The hook wraps
    model.model.diffusion_model.final_layer so forecast outputs replace the
    full DiT forward while the cheap final projection still runs normally.
    stats: optional dict receiving {'actual', 'forecast', 'reason'}.
    """
    out = {'actual': 0, 'forecast': 0, 'reason': 'disabled'}
    if stats is None:
        stats = out
    else:
        stats.update(out)
    if not enabled:
        yield stats
        return
    if float(cfg or 1.0) <= 1.0:
        stats['reason'] = 'cfg_1_no_uncond'
        yield stats
        return
    total = max(1, int(steps))
    if total < 8:
        stats['reason'] = 'too_few_steps'
        yield stats
        return
    try:
        diffusion = model.model.diffusion_model
        final_layer = diffusion.final_layer
        original = final_layer.forward
    except AttributeError:
        stats['reason'] = 'unsupported_model'
        yield stats
        return

    import torch

    warmup = max(5, total // 4)
    min_history = 4
    final_steps = max(2, total // 5)
    # Stride 3 over stride 2: the reference build measured stride 2 as
    # prompt-dependent (excellent on some prompts, merged fingers on others),
    # which is disqualifying for a default. Slower but prompt-agnostic.
    stride = 3
    # Separate history and call counter per CFG branch (cond/uncond); the
    # branches have different feature statistics and must never share a fit.
    branches = {0: {'history': [], 'call': 0}, 1: {'history': [], 'call': 0}}
    stats['reason'] = 'active'

    def _basis(values, degree=2):
        x = torch.tensor(values, dtype=torch.float32)
        columns = [torch.ones_like(x), x]
        for _ in range(2, degree + 1):
            columns.append(2 * x * columns[-1] - columns[-2])
        return torch.stack(columns[: degree + 1], dim=1)

    def _forecast(history, coord, blend=0.5):
        coords = [item[0] for item in history]
        design = _basis(coords)
        gram = design.T @ design + 0.1 * torch.eye(design.shape[1])
        phi = _basis([coord])
        spectral = (phi @ torch.linalg.solve(gram, design.T)).flatten()
        spacing = coords[-1] - coords[-2]
        ratio = 0.0 if abs(spacing) < 1e-12 else (coord - coords[-1]) / spacing
        linear_w = torch.zeros(len(history), dtype=torch.float32)
        linear_w[-2], linear_w[-1] = -ratio, 1.0 + ratio
        weights = blend * spectral + (1.0 - blend) * linear_w
        weights[-1] += 1.0 - weights.sum()
        result = torch.zeros_like(history[-1][1], dtype=torch.float32)
        for weight, (_, feature) in zip(weights.tolist(), history):
            result.add_(feature.float(), alpha=weight)
        return result.to(history[-1][1].dtype)

    index = {'call': 0}

    def wrapped(*args, **kwargs):
        if stop is not None and stop():
            raise InterruptedError('Generation interrupted')
        # MethodType(wrapped, final_layer) makes wrapped receive the module as
        # args[0]; `original` is likewise bound, so forward only the remaining
        # call arguments to it.
        call_args, call_kwargs = args[1:], dict(kwargs)
        # The feature is the first tensor argument. Remember its position so a
        # forecast can replace it in place without disturbing the signature.
        feature = None
        feature_index = None
        for position, candidate in enumerate(call_args):
            if torch.is_tensor(candidate):
                feature = candidate
                feature_index = position
                break
        if feature is None:
            for name in ('x', 'hidden_states'):
                if torch.is_tensor(kwargs.get(name)):
                    feature = kwargs[name]
                    break
        call = index['call']
        index['call'] += 1
        # At CFG > 1 ComfyUI runs two forwards per step (cond, then uncond).
        # The two branches have different feature statistics; mixing them
        # corrupts the fit. Parity identifies the branch because the sampler
        # visits steps in order with a fixed two-pass pattern. All warmup,
        # final-step and stride logic counts in STEP units, per branch.
        branch = call % 2
        step = call // 2
        own = branches[branch]
        history = own['history']
        own_call = own['call']
        own['call'] += 1
        actual = (
            step < warmup
            or step >= total - final_steps
            or own_call % stride != 0
            or len(history) < min_history
            or feature is None
            or (history and tuple(history[-1][1].shape) != tuple(feature.shape))
        )
        if not actual:
            predicted = _forecast(history, float(step))
            latest = history[-1][1].float()
            change = (predicted.float() - latest).square().mean().sqrt()
            limit = latest.square().mean().sqrt().clamp_min(1e-6) * min(0.25, 0.5 / max(1.0, float(cfg)))
            if torch.isfinite(predicted).all() and change <= limit:
                stats['forecast'] += 1
                if feature_index is not None:
                    call_args = call_args[:feature_index] + (predicted.to(feature.dtype),) + call_args[feature_index + 1:]
                else:
                    call_kwargs['x'] = predicted.to(feature.dtype)
                return original(*call_args, **call_kwargs)
        history.append((float(step), feature.detach().to('cpu').contiguous()))
        del history[:-min_history]
        stats['actual'] += 1
        return original(*call_args, **call_kwargs)

    final_layer.forward = types.MethodType(wrapped, final_layer)
    try:
        yield stats
    finally:
        final_layer.forward = original
        branches.clear()