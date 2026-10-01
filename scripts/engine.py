"""Normal checkpoint dropdown + Generate. No selectable script or additional tab."""
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
                with gr.Accordion('Models', open=False):
                    gr.Markdown(
                        'Put the Ming-Image DiT, Ling text encoder and VAE in `models/Ming-Image` '
                        '(or `diffusion_models` / `text_encoders` / `VAE`). '
                        'Downloads: [Comfy-Org/Ming-Image](https://huggingface.co/Comfy-Org/Ming-Image). '
                        'Read the model license before downloading.')
                with gr.Accordion('Speed', open=False):
                    spectrum = gr.Checkbox(label='Spectrum acceleration', value=False,
                                           info='Approximate CFG pass prediction. Real gains at CFG above 1; '
                                                'at CFG 1 there is nothing to skip. Turn off for an exact baseline.')
                with gr.Accordion('Optional settings', open=False):
                    rgba = gr.Checkbox(label='RGBA (transparent background)', value=False,
                                       info='Save RGBA output when the prompt requests a transparent background.')
                    memory = gr.Dropdown(label='Memory mode', choices=['auto', 'lowvram', 'cpu'], value='auto',
                                         info='lowvram reduces GPU residency; cpu is a slow fallback.')
                    keep = gr.Checkbox(label='Keep model in memory', value=False,
                                       info='Reuse the worker between runs. Selecting another preset releases it.')
                    lora = gr.Dropdown(label='LoRA', choices=['(none)'], value='(none)', allow_custom_value=True,
                                       info='Or use <lora:filename:strength> prompt tags.')
                    strength = gr.Slider(label='LoRA strength', minimum=-2.0, maximum=2.0, step=0.01, value=1.0)
                refs = []
                if is_img2img:
                    with gr.Accordion('Extra reference images', open=False):
                        for n in range(2, 9):
                            refs.append(gr.Image(label=f'Reference {n}', type='pil', interactive=True))
            forge.register_ui_binding(panel, is_img2img)
            # Order must match pi_ming.forge.KEYS exactly, then reference images.
            return [rgba, memory, keep, lora, strength, spectrum, *refs]

    forge.install()
    install_preset()
except (ImportError, AttributeError) as error:
    print(f'[PI-Ming] Forge UI unavailable: {error}')