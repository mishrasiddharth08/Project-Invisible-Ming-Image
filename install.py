"""Verify the extension is complete. Ming-Image reuses Forge's Python and the
pinned vendored ComfyUI backend; nothing is installed into Forge's environment."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
REQUIRED = [
    ROOT / 'pi_ming' / 'worker.py',
    ROOT / 'pi_ming' / 'forge.py',
    ROOT / 'scripts' / 'engine.py',
    ROOT / 'resources' / 'catalog.json',
    ROOT / 'vendor' / 'ComfyUI' / 'comfy' / 'supported_models.py',
]


def main():
    missing = [str(p) for p in REQUIRED if not p.is_file()]
    if missing:
        print('[PI-Ming] Incomplete extension; missing files:')
        for path in missing:
            print('  ' + path)
        sys.exit(1)
    print('[PI-Ming] Extension complete. Select the Ming-Image preset after restarting Forge.')


if __name__ == '__main__':
    main()