"""Fetch the pinned vendored ComfyUI backend after cloning.

The ComfyUI checkout is gitignored (large, GPL-licensed separately), so this
script clones it at a pinned revision into vendor/ComfyUI. Run once:

    python fetch_vendor.py

Requires git. The pin is a full commit SHA; change it deliberately and test.
"""
import subprocess
import sys
from pathlib import Path

REPO = 'https://github.com/comfyanonymous/ComfyUI'
PIN = 'REPLACE-WITH-THE-COMMIT-SHA-YOUR-MING-PR-BASELINE-USES'
TARGET = Path(__file__).resolve().parent / 'vendor' / 'ComfyUI'


def main():
    if TARGET.is_dir() and any(TARGET.iterdir()):
        print('[PI-Ming] vendor/ComfyUI already present; delete it first to re-fetch.')
        return
    if not PIN.startswith(tuple('0123456789abcdef')) or len(PIN) != 40:
        raise SystemExit('[PI-Ming] Set the pinned commit SHA in fetch_vendor.py before use.')
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(['git', 'clone', REPO, str(TARGET)])
    subprocess.check_call(['git', '-C', str(TARGET), 'checkout', '--detach', PIN])
    print('[PI-Ming] vendored ComfyUI pinned at ' + PIN)


if __name__ == '__main__':
    main()