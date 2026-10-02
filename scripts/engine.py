"""Normal checkpoint dropdown + Generate. No selectable script or additional tab.

The Ming panel is deliberately minimal: Forge's own Spectrum extension
(extensions-builtin/sd_forge_spectrum) accelerates sampling when enabled,
LoRAs come from <lora:...> prompt tags, memory decisions are automatic, and
model reuse follows Forge's standard model lifecycle.
"""
from pathlib import Path
import sys

EXT = Path(__file__).resolve().parent.parent
if str(EXT) not in sys.path:
    sys.path.insert(0, str(EXT))

from pi_ming import forge
from pi_ming.preset import install as install_preset

try:
    import gradio as gr
    from modules import scripts, script_callbacks

    class Script(scripts.Script):
        is_ming_invisible = True

        def title(self):
            return 'PROJECT INVISIBLE — Ming-Image 0.1 Design'

        def show(self, is_img2img):
            return scripts.AlwaysVisible

        def ui(self, is_img2img):
            prefix = 'ming_img2img' if is_img2img else 'ming_txt2img'
            with gr.Accordion('Ming', open=False, visible=False, elem_id=prefix) as panel:
                gr.Markdown(
                    'Ming-Image runs with the native controls: 1024/2048 square, '
                    '12 steps, CFG 1.0 recommended. Use `<lora:filename:strength>` '
                    'in the prompt for LoRAs, and Forge\'s Spectrum extension for '
                    'acceleration. RGBA output follows the prompt\'s '
                    'transparent-background phrasing.')
                rgba = gr.Checkbox(label='RGBA (transparent background)', value=False,
                                   info='Save RGBA output when the prompt requests a transparent background.')
                refs = []
                if is_img2img:
                    with gr.Accordion('Extra reference images', open=False):
                        for n in range(2, 9):
                            refs.append(gr.Image(label=f'Reference {n}', type='pil', interactive=True))
            forge.register_ui_binding(panel, is_img2img)
            # Order must match pi_ming.forge.KEYS exactly, then reference images.
            return [rgba, *refs]

    forge.install()
    install_preset()
except (ImportError, AttributeError) as error:
    print(f'[PI-Ming] Forge UI unavailable: {error}')