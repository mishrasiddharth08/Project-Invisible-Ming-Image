"""Adapted from H3 legacy_preset.py; additive native preset registration."""
from copy import copy, deepcopy
from .forge import LABEL

NAME = 'Ming-Image'


def install():
    from modules import shared
    from modules_forge import presets
    template = {}
    presets.register(template)
    ours = {}
    for key, info in template.items():
        if key.startswith('sd_'):
            target = NAME + key[2:]
        elif key in ('forge_checkpoint_sd', 'forge_additional_modules_sd', 'forge_unet_storage_dtype_sd'):
            target = key[:-2] + NAME
        else:
            continue
        item = copy(info)
        item.default = deepcopy(info.default)
        if getattr(info, 'section', (None,))[0] == 'ui_sd':
            item.section = ('ui_ming', 'Ming-Image 0.1')
        if target.endswith('_sampler'):
            item.default = 'Res Multistep'
        elif target.endswith('_scheduler'):
            item.default = 'Normal'
        elif target.endswith('_step'):
            item.default = 12
        elif target.endswith('_cfg'):
            item.default = 1.0
        elif target.endswith(('_width', '_height')):
            item.default = 1024
            # Reference buckets are 1024 and 2048; hard limits 256–4096 in multiples of 16.
            item.component_args = {'minimum': 256, 'maximum': 4096, 'step': 16}
        elif target == 'forge_checkpoint_' + NAME:
            item.default = LABEL
        elif target == 'forge_additional_modules_' + NAME:
            item.default = []
        ours[target] = item
    required = {'forge_checkpoint_' + NAME, NAME + '_t2i_step', NAME + '_t2i_cfg'}
    if not required.issubset(ours):
        raise RuntimeError('Neo preset schema changed; Ming-Image preset registration skipped')
    for key, info in ours.items():
        if key not in shared.opts.data_labels:
            shared.opts.add_option(key, info)
    original = presets.PresetArch.choices
    if not getattr(original, '_pi_ming', False):
        def choices(*a, **kw):
            values = list(original(*a, **kw))
            if NAME not in values:
                values.append(NAME)
            return values
        choices._pi_ming = True
        presets.PresetArch.choices = staticmethod(choices)