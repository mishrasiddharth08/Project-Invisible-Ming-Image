from pathlib import Path
import threading
from .config import ROOT, LABEL, PRESET
from . import runtime
from .assets import classify, scan

# Must match scripts/engine.py ui() return order exactly.
KEYS = ('rgba', 'memory', 'keep_loaded', 'lora', 'strength', 'spectrum')
REF_COUNT = 7
_UI_BINDINGS = []
_selection_state = threading.local()


def selected(p=None):
    from modules import shared, sd_models
    value = (getattr(p, 'override_settings', {}) or {}).get('sd_model_checkpoint', shared.opts.sd_model_checkpoint)
    return selected_value(value)


def selected_value(value):
    from modules import sd_models
    item = sd_models.checkpoints_list.get(value) or sd_models.checkpoint_aliases.get(value)
    return value == LABEL or bool(getattr(item, '_pi_ming', False)) or classify(getattr(item, 'filename', '')) in ('dit', 'dit_layer')


def _checkpoint_identity(value):
    from modules import sd_models
    item = sd_models.checkpoints_list.get(value) or sd_models.checkpoint_aliases.get(value)
    filename = getattr(item, 'filename', None)
    return str(Path(filename).resolve()).casefold() if filename else str(value or '').casefold()


def _same_checkpoint(left, right):
    return _checkpoint_identity(left) == _checkpoint_identity(right)


def _cancel(reason):
    if runtime._worker is not None:
        print(f'[PI-Ming] cancelling dedicated worker: {reason}')
    runtime.selection_changed()


def register():
    from modules import sd_models
    if LABEL in sd_models.checkpoints_list:
        return
    ci = object.__new__(sd_models.CheckpointInfo)
    # A real Ming checkpoint keeps strict checkpoint managers from trying to
    # fingerprint the metadata-only fallback marker.
    try:
        inventory = scan()
    except ImportError:
        inventory = {'dit': []}
    candidates = inventory.get('dit') or inventory.get('dit_layer') or []
    ci.filename = candidates[0] if candidates else str(ROOT / 'resources' / 'ming.json')
    ci.name = ci.title = ci.short_title = ci.model_name = ci.name_for_extra = LABEL
    ci.hash = 'pi-ming'
    ci.sha256 = ci.shorthash = None
    ci.metadata = {}
    ci.is_safetensors = False
    ci._pi_ming = True
    ci.ids = [LABEL, ci.filename]
    ci.calculate_shorthash = lambda: None
    ci.register()


def options(runner, p):
    result = {}
    for script in getattr(runner, 'alwayson_scripts', []) or []:
        if getattr(script, '_pi_ming', False):
            values = list(getattr(p, 'script_args', []) or [])[script.args_from:script.args_to]
            result = {**dict(zip(KEYS, values[:len(KEYS)])), 'refs': values[len(KEYS):len(KEYS)+REF_COUNT]}
            break
    from modules import shared, sd_models
    value = (getattr(p, 'override_settings', {}) or {}).get('sd_model_checkpoint', shared.opts.sd_model_checkpoint)
    item = sd_models.checkpoints_list.get(value) or sd_models.checkpoint_aliases.get(value)
    filename = getattr(item, 'filename', '')
    if not getattr(item, '_pi_ming', False) and classify(filename) in ('dit', 'dit_layer'):
        result['dit_layer' if classify(filename) == 'dit_layer' else 'dit'] = filename
    # Native "VAE / Text Encoder" multi-dropdown wins over the hidden extension selectors.
    for module in getattr(shared.opts, 'forge_additional_modules', []) or []:
        role = classify(str(module))
        if role in ('clip', 'clip_layer', 'vae'):
            result[role] = str(module)
    return result


def install_selection():
    from modules import shared
    for key in ('sd_model_checkpoint', 'forge_preset'):
        item = shared.opts.data_labels.get(key)
        if item is None or getattr(item.onchange, '_pi_ming', False):
            continue
        previous = item.onchange
        def changed(*args, previous=previous, key=key, **kwargs):
            if (key == 'sd_model_checkpoint' and getattr(shared.opts, 'forge_preset', None) == PRESET
                    and not selected() and (getattr(_selection_state, 'ming_callback', False)
                                            or getattr(_selection_state, 'ming_refresh', False))
                    and not getattr(_selection_state, 'explicit_leave', False)):
                # Some older extension callbacks repair their own saved model
                # while Forge has temporarily cleared the checkpoint registry.
                # Ming's preset remains authoritative during that refresh.
                shared.opts.set('sd_model_checkpoint', LABEL, run_callbacks=False)
                shared.opts.set('forge_checkpoint_' + PRESET, LABEL, run_callbacks=False)
                print('[PI-Ming] ignored transient foreign checkpoint restore while preset Ming-Image is active')
                return None
            if ((key == 'forge_preset' and shared.opts.forge_preset != PRESET)
                    or (key == 'sd_model_checkpoint' and not selected()
                        and not getattr(_selection_state, 'explicit_leave', False))):
                _cancel(f'option changed: {key}')
            if previous:
                nested_guard = (key == 'sd_model_checkpoint'
                                and getattr(shared.opts, 'forge_preset', None) == PRESET
                                and selected())
                old_guard = getattr(_selection_state, 'ming_callback', False)
                if nested_guard:
                    _selection_state.ming_callback = True
                try:
                    return previous(*args, **kwargs)
                finally:
                    _selection_state.ming_callback = old_guard
        changed._pi_ming = True
        shared.opts.onchange(key, changed, call=False)


def restore_saved_selection():
    """Honor Forge's saved Ming-Image preset after other extensions restore theirs."""
    from modules import shared
    if getattr(shared.opts, 'forge_preset', None) != PRESET:
        return
    current = getattr(shared.opts, 'sd_model_checkpoint', None)
    checkpoint = current if selected_value(current) else LABEL
    shared.opts.set('sd_model_checkpoint', checkpoint)
    shared.opts.set('forge_checkpoint_' + PRESET, checkpoint)
    shared.opts.set('forge_additional_modules', [])
    shared.opts.set('forge_additional_modules_' + PRESET, [])


def _wrap_process(original):
    if getattr(original, '_pi_ming_process', False):
        return original
    def process(p, *args, **kwargs):
        if selected(p):
            return runtime.generate(p, options(getattr(p, 'scripts', None), p))
        _cancel('ordinary process_images request')
        return original(p, *args, **kwargs)
    process._pi_ming_process = True
    process._pi_ming_original = original
    return process


# Every wrapped host function is recorded so unloading this extension restores
# exactly what was there before — including wrappers other extensions installed
# after ours. We never unwrap someone else's wrapper.
_RESTORE = []


def _install_attribute(owner, name, replacement, tag):
    original = getattr(owner, name, None)
    if not callable(original) or getattr(original, tag, False):
        return False
    setattr(owner, name, replacement(original))
    _RESTORE.append((owner, name, original))
    return True


def _restore_all():
    while _RESTORE:
        owner, name, original = _RESTORE.pop()
        try:
            setattr(owner, name, original)
        except Exception as error:
            print(f'[PI-Ming] could not restore {name}: {error}')
    global _UI_BINDINGS
    _UI_BINDINGS = []


def install_forge_selection():
    """Bypass stock checkpoint loading only for the worker-owned Ming marker."""
    try:
        from modules import shared
        from modules_forge import main_entry
    except ImportError:
        return
    original = getattr(main_entry, 'checkpoint_change', None)
    if not callable(original) or getattr(original, '_pi_ming_checkpoint', False):
        return
    def checkpoint_change(ckpt_name, preset, save=True, refresh=True):
        if not selected_value(ckpt_name):
            if selected():
                _cancel(f'checkpoint leaving Ming-Image: {ckpt_name!r}')
            _selection_state.explicit_leave = True
            try:
                return original(ckpt_name, preset, save=save, refresh=refresh)
            finally:
                _selection_state.explicit_leave = False
        current = getattr(shared.opts, 'sd_model_checkpoint', None)
        changed = not _same_checkpoint(current, ckpt_name)
        effective = ckpt_name if changed else current
        shared.opts.set('sd_model_checkpoint', effective)
        if preset is not None:
            shared.opts.set('forge_checkpoint_' + preset, effective)
        shared.opts.set('forge_additional_modules', [])
        if preset is not None:
            shared.opts.set('forge_additional_modules_' + preset, [])
        if save:
            shared.opts.save(shared.config_filename)
        if changed:
            _cancel(f'Ming checkpoint changed: {current!r} -> {ckpt_name!r}')
        return changed
    checkpoint_change._pi_ming_checkpoint = True
    checkpoint_change._pi_ming_original = original
    main_entry.checkpoint_change = checkpoint_change


def register_ui_binding(panel, is_img2img=False):
    binding = (panel, bool(is_img2img))
    if binding not in _UI_BINDINGS:
        _UI_BINDINGS.append(binding)


def _ui_load_state(has_denoise):
    import gradio as gr
    from modules import shared
    if getattr(shared.opts, 'forge_preset', None) == PRESET and not selected():
        restore_saved_selection()
    value = getattr(shared.opts, 'sd_model_checkpoint', None)
    active = getattr(shared.opts, 'forge_preset', None) == PRESET and selected_value(value)
    updates = [gr.update(value=value), gr.update(value=[]) if active else gr.skip(), gr.update(visible=active, open=False)]
    if has_denoise:
        updates.append(gr.update(value=1.0) if active else gr.skip())
    return updates


def _ui_selection_state(value, has_denoise):
    import gradio as gr
    from modules import shared
    active = selected_value(value)
    backend_active = (getattr(shared.opts, 'forge_preset', None) == PRESET and selected())
    # Page initialization can briefly emit the checkpoint embedded before Ming
    # restores its saved state. Never cancel an active Ming request for that
    # stale browser value; a real switch is cancelled by the backend option hook.
    if not active and not backend_active:
        _cancel(f'UI selected non-Ming checkpoint: {value!r}')
    updates = [gr.update(visible=active, open=False), gr.update(value=[]) if active else gr.skip()]
    if has_denoise:
        updates.append(gr.update(value=1.0) if active else gr.skip())
    return updates


def install_forge_ui_sync():
    """Register after Forge's own page-load preset callback, so Ming wins last."""
    try:
        from modules_forge import main_entry
    except ImportError:
        return
    original = getattr(main_entry, 'forge_main_entry', None)
    if not callable(original) or getattr(original, '_pi_ming_ui_sync', False):
        return
    def forge_main_entry(*args, **kwargs):
        result = original(*args, **kwargs)
        from gradio.context import Context
        checkpoint, modules = main_entry.ui_checkpoint, main_entry.ui_vae
        for panel, is_img2img in list(_UI_BINDINGS):
            denoise = main_entry.get_a1111_ui_component('img2img', 'Denoising strength') if is_img2img else None
            change_outputs = [panel, modules] + ([denoise] if denoise is not None else [])
            checkpoint.change(
                lambda value, has_denoise=denoise is not None: _ui_selection_state(value, has_denoise),
                inputs=[checkpoint], outputs=change_outputs, queue=False, show_progress='hidden')
            outputs = [checkpoint, modules, panel] + ([denoise] if denoise is not None else [])
            Context.root_block.load(
                lambda has_denoise=denoise is not None: _ui_load_state(has_denoise),
                outputs=outputs, queue=False, show_progress='hidden')
        return result
    forge_main_entry._pi_ming_ui_sync = True
    forge_main_entry._pi_ming_original = original
    main_entry.forge_main_entry = forge_main_entry


def _ready(*args, **kwargs):
    import sys
    from modules import processing
    def _process_wrapper(original):
        def process(p, *a, **kw):
            if selected(p):
                return runtime.generate(p, options(getattr(p, 'scripts', None), p))
            _cancel('ordinary process_images request')
            return original(p, *a, **kw)
        process._pi_ming_process = True
        return process
    _install_attribute(processing, 'process_images', _process_wrapper, '_pi_ming_process')
    for name in ('modules.api.api', 'modules.txt2img', 'modules.img2img'):
        module = sys.modules.get(name)
        if module is not None and callable(getattr(module, 'process_images', None)):
            _install_attribute(module, 'process_images', _process_wrapper, '_pi_ming_process')
    install_selection()
    install_forge_selection()
    install_forge_ui_sync()
    if not args and not kwargs:
        _UI_BINDINGS.clear()
    restore_saved_selection()
    register()


def _ensure_callback(script_callbacks, category, register_cb, callback):
    callback_map = getattr(script_callbacks, 'callback_map', None)
    current = callback_map.get(category, []) if isinstance(callback_map, dict) else []
    if any(getattr(item, 'callback', None) is callback for item in current):
        return
    register_cb(callback)


def install():
    from modules import scripts, processing, sd_models, script_callbacks
    if not all(callable(x) for x in (getattr(scripts.ScriptRunner, 'run', None),
            getattr(processing, 'process_images', None), getattr(sd_models, 'list_models', None))):
        # Fail loudly instead of patching a changed host half-blindly.
        raise RuntimeError('Forge hooks changed; Ming-Image disabled. Other engines are unchanged.')
    if not getattr(scripts.ScriptRunner.run, '_pi_ming', False):
        def run_wrapper(original_run):
            def run(runner, p, *args, **kwargs):
                if selected(p):
                    return runtime.generate(p, options(runner, p))
                _cancel('ordinary ScriptRunner request')
                return original_run(runner, p, *args, **kwargs)
            run._pi_ming = True
            return run
        def listing_wrapper(original_list):
            def listing(*args, **kwargs):
                from modules import shared
                guard = (getattr(shared.opts, 'forge_preset', None) == PRESET and selected())
                old_guard = getattr(_selection_state, 'ming_refresh', False)
                if guard:
                    _selection_state.ming_refresh = True
                try:
                    result = original_list(*args, **kwargs)
                finally:
                    register()
                    _selection_state.ming_refresh = old_guard
                restore_saved_selection()
                return result
            listing._pi_ming = True
            return listing
        _install_attribute(scripts.ScriptRunner, 'run', run_wrapper, '_pi_ming')
        _install_attribute(sd_models, 'list_models', listing_wrapper, '_pi_ming')
    _install_attribute(processing, 'process_images', _wrap_process, '_pi_ming_process')
    _ensure_callback(script_callbacks, 'callbacks_app_started', script_callbacks.on_app_started, _ready)
    before_ui = getattr(script_callbacks, 'on_before_ui', None)
    if callable(before_ui):
        _ensure_callback(script_callbacks, 'callbacks_before_ui', before_ui, _ready)
    _ensure_callback(script_callbacks, 'callbacks_script_unloaded', script_callbacks.on_script_unloaded, _on_unload)
    install_forge_selection()
    install_forge_ui_sync()
    register()


def _on_unload(*args, **kwargs):
    """Extension removed/reloaded: release the worker and restore every hook."""
    runtime.selection_changed()
    _restore_all()