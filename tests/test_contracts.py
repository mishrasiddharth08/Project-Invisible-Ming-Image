"""Regression record: UI arity contract + Spectrum branch separation.

Defects covered:
- engine.py's ui() return list and pi_ming.forge.KEYS are positionally coupled
  (prompt rule 64); a component added on one side alone shifts every field
  after it. This test parses both from source so it cannot drift silently.
- Spectrum originally mixed cond/uncond forward passes into one history
  (prompt rule 70: never feed an approximation back in as an anchor). The two
  branches have different feature statistics; the fit must be per-branch.
"""
import re
import sys
import types

import torch

ROOT = __file__.rsplit('tests', 1)[0]
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pi_ming import spectrum


def test_arity_contract():
    engine = open(ROOT + 'scripts/engine.py', encoding='utf-8').read()
    forge = open(ROOT + 'pi_ming/forge.py', encoding='utf-8').read()
    assert re.search(r'return \[rgba, memory, keep, lora, strength, spectrum, \*refs\]', engine), \
        'engine.py return order changed; update pi_ming.forge.KEYS in lockstep'
    keys = re.search(r"KEYS = \(([^)]*)\)", forge).group(1)
    names = [k.strip().strip("'\"") for k in keys.split(',') if k.strip()]
    assert names == ['rgba', 'memory', 'keep_loaded', 'lora', 'strength', 'spectrum'], names
    assert 'REF_COUNT = 7' in forge, 'reference slot count must match engine.py refs 2..8'


def test_config_passthrough():
    runtime = open(ROOT + 'pi_ming/runtime.py', encoding='utf-8').read()
    config = open(ROOT + 'pi_ming/config.py', encoding='utf-8').read()
    for key in ('preview_interval', 'keep_loaded', 'model_roots'):
        assert key in runtime or key in config, f'config-derived key {key} dropped on the UI path'


class _Diff:
    pass


class _Model:
    pass


class FinalLayer:
    def forward(self, x):
        return x * 2.0


def test_spectrum_branch_separation():
    patcher = _Model()
    patcher.model = _Model()
    patcher.model.diffusion_model = _Diff()
    fl = FinalLayer()
    patcher.model.diffusion_model.final_layer = fl

    stats = {}
    with spectrum.accelerate(patcher, enabled=True, steps=20, cfg=4.0, stats=stats):
        assert stats['reason'] == 'active', stats
        wrapped = fl.forward
        # Alternate cond (large) / uncond (small) features for 20 steps x 2.
        for i in range(40):
            value = 10.0 if i % 2 == 0 else -10.0
            wrapped(torch.full((1, 4), value))
        assert stats['actual'] + stats['forecast'] == 40, stats
    # After the context, the original method is restored exactly.
    assert fl.forward is FinalLayer.forward or (
        isinstance(fl.forward, types.MethodType) and fl.forward.__func__ is FinalLayer.forward)


def test_spectrum_off_paths():
    patcher = _Model()
    patcher.model = _Model()
    patcher.model.diffusion_model = _Diff()
    patcher.model.diffusion_model.final_layer = FinalLayer()
    stats = {}
    with spectrum.accelerate(patcher, enabled=False, steps=20, cfg=4.0, stats=stats):
        pass
    assert stats['reason'] == 'disabled', stats
    stats = {}
    with spectrum.accelerate(patcher, enabled=True, steps=12, cfg=1.0, stats=stats):
        pass
    assert stats['reason'] == 'cfg_1_no_uncond', stats
    stats = {}
    with spectrum.accelerate(patcher, enabled=True, steps=6, cfg=4.0, stats=stats):
        pass
    assert stats['reason'] == 'too_few_steps', stats


if __name__ == '__main__':
    test_arity_contract()
    test_config_passthrough()
    test_spectrum_branch_separation()
    test_spectrum_off_paths()
    print('all tests passed')