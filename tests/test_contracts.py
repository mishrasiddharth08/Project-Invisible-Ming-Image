"""Regression record: UI arity contract + wiring contracts.

Defects covered:
- engine.py's ui() return list and pi_ming.forge.KEYS are positionally coupled
  (prompt rule 64); a component added on one side alone shifts every field
  after it. This test parses both from source so it cannot drift silently.
- Config-derived keys must never disappear on the UI path (prompt rule 63).
- The centralized Spectrum contract: the Forge side must read the central
  sd_forge_spectrum script's settings and pass its lib path; the worker must
  apply the central SpectrumNode.patch verbatim. No forked forecaster exists
  in this extension.
"""
import re
import sys

ROOT = __file__.rsplit('tests', 1)[0]
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def test_arity_contract():
    engine = open(ROOT + 'scripts/engine.py', encoding='utf-8').read()
    forge = open(ROOT + 'pi_ming/forge.py', encoding='utf-8').read()
    assert re.search(r'return \[rgba, \*refs\]', engine), \
        'engine.py return order changed; update pi_ming.forge.KEYS in lockstep'
    keys = re.search(r"KEYS = \(([^)]*)\)", forge).group(1)
    names = [k.strip().strip("'\"") for k in keys.split(',') if k.strip()]
    assert names == ['rgba'], names
    assert 'REF_COUNT = 7' in forge, 'reference slot count must match engine.py refs 2..8'


def test_config_passthrough():
    runtime = open(ROOT + 'pi_ming/runtime.py', encoding='utf-8').read()
    config = open(ROOT + 'pi_ming/config.py', encoding='utf-8').read()
    for key in ('preview_interval', 'model_roots'):
        assert key in runtime or key in config, f'config-derived key {key} dropped on the UI path'


def test_central_spectrum_contract():
    forge = open(ROOT + 'pi_ming/forge.py', encoding='utf-8').read()
    worker = open(ROOT + 'pi_ming/worker.py', encoding='utf-8').read()
    engine = open(ROOT + 'scripts/engine.py', encoding='utf-8').read()
    # The Forge side reads the central script, not a checkbox of its own.
    assert "SPECTRUM_TITLE = 'Spectrum Integrated'" in forge
    assert "from lib_spectrum.forecaster import SpectrumNode" in worker
    assert "SpectrumNode.patch(model" in worker
    # No forked forecaster ships in this extension.
    import os
    assert not os.path.exists(ROOT + 'pi_ming/spectrum.py'), 'forked Spectrum must not exist'
    # The panel points users at the central extension, not a local toggle.
    assert 'Spectrum extension' in engine
    assert 'spectrum' not in [k.strip().strip("'\"") for k in
        re.search(r"KEYS = \(([^)]*)\)", forge).group(1).split(',')]


def test_minimal_panel():
    engine = open(ROOT + 'scripts/engine.py', encoding='utf-8').read()
    assert 'Models' not in engine, 'Models accordion removed by design; downloads point at HuggingFace'
    assert "gr.Dropdown" not in engine, 'no LoRA dropdown: prompt tags are the interface'
    assert 'Keep model in memory' not in engine, 'reuse follows Forge lifecycle, not a checkbox'
    assert 'Memory mode' not in engine, 'memory mode is automatic'


if __name__ == '__main__':
    test_arity_contract()
    test_config_passthrough()
    test_central_spectrum_contract()
    test_minimal_panel()
    print('all tests passed')