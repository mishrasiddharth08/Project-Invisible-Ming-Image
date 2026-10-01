"""Validate inexpensive request settings before starting the model worker."""
import math
import re

# Lumina2-family flow samplers exposed by the ComfyUI backend; Automatic maps to res_multistep.
SAMPLERS = {'res_multistep': 'res_multistep', 'euler': 'euler', 'deis': 'deis', 'dpmpp_2m': 'dpmpp_2m',
            'dpmpp_2m_sde': 'dpmpp_2m_sde', 'dpmpp_3m_sde': 'dpmpp_3m_sde', 'ddim': 'ddim', 'euler_ancestral': 'euler_ancestral'}
SCHEDULERS = {'simple': 'simple', 'normal': 'normal', 'beta': 'beta', 'shift': 'shift', 'exponential': 'exponential'}
DISPLAY_SAMPLERS = {'Res Multistep': 'res_multistep', 'Euler': 'euler', 'DEIS': 'deis', 'DPM++ 2M': 'dpmpp_2m',
                    'DPM++ 2M SDE': 'dpmpp_2m_sde', 'DPM++ 3M SDE': 'dpmpp_3m_sde', 'DDIM': 'ddim',
                    'Euler ancestral': 'euler_ancestral'}


def validate(width, height, steps, cfg, sampler, scheduler, editing=False, image=None, mask=None,
             denoise=1.0, hires=False, rgba=False):
    # Ming-Image uses 1024/2048 square buckets for its native text-to-image path.
    if any(int(v) != v or not 256 <= v <= 4096 or v % 16 for v in (width, height)):
        raise ValueError('Ming-Image width and height must be multiples of 16, between 256 and 4096.')
    if width != height:
        print('[PI-Ming] Note: the model is validated on square 1024/2048 buckets; non-square sizes may drift from reference quality.')
    if int(steps) != steps or not 1 <= steps <= 100:
        raise ValueError('Choose 1–100 steps; the reference workflow uses 12.')
    if not math.isfinite(cfg) or not 0.5 <= cfg <= 20:
        raise ValueError('Ming-Image CFG must be between 0.5 and 20; the reference workflow uses 1.0.')
    if sampler not in SAMPLERS and sampler not in DISPLAY_SAMPLERS and sampler != 'Automatic':
        print('[PI-Ming] Unsupported sampler "' + str(sampler) + '" ignored; using Res Multistep.')
        sampler = 'Res Multistep'
    scheduler_name = str(scheduler).casefold()
    if scheduler_name not in SCHEDULERS and scheduler_name != 'automatic':
        print('[PI-Ming] Unsupported scheduler "' + str(scheduler) + '" ignored; using Normal.')
        scheduler_name = 'normal'
    if editing and image is None:
        raise ValueError('Add an image in img2img before generating an edit.')
    if mask is not None:
        raise ValueError('Ming-Image reference editing does not support masks. Use plain img2img.')
    if editing and abs(float(denoise) - 1.0) > 1e-6:
        raise ValueError('Set img2img Denoising strength to 1. Ming-Image uses reference conditioning, not SD denoising.')
    if hires:
        raise ValueError('Disable Hires fix for Ming-Image. Set the desired final width and height directly.')
    if rgba:
        print('[PI-Ming] RGBA requested: transparent-background prompts use the model\'s recommended RGBA phrasing.')
    return SAMPLERS.get(DISPLAY_SAMPLERS.get(sampler, sampler), sampler), SCHEDULERS.get(scheduler_name, 'normal')


def parse_loras(prompt):
    adapters = []
    def take(match):
        name, raw_strength = match.group(1), match.group(2)
        try:
            strength = float(raw_strength)
        except ValueError:
            raise ValueError('Invalid LoRA strength: <lora:' + name + ':' + raw_strength + '>. Use a number between -2 and 2.') from None
        if not math.isfinite(strength) or not -2 <= strength <= 2:
            raise ValueError('LoRA strength must be between -2 and 2.')
        adapters.append((name, strength))
        return ''
    prompt = re.sub(r'<lora:([^:<>]+):([-+\d.eE]+)>', take, prompt)
    if re.search(r'<(?:lora|lyco|hypernet):', prompt, re.I):
        raise ValueError('Unsupported or malformed adapter tag. Use <lora:Ming-filename:strength>.')
    return prompt.strip(), adapters