"""Launch Isaac Lab with the parity evaluator's Isaac Sim experience."""

import os
from pathlib import Path


def launch(render=True):
    from isaaclab.app import AppLauncher

    root = Path(os.environ["EXP_PATH"])
    candidates = ("omni.isaac.sim.python.kit", "isaacsim.exp.base.python.kit", "isaacsim.exp.base.kit")
    experience = next((root / name for name in candidates if (root / name).is_file()), None)
    if experience is None:
        raise FileNotFoundError(f"No Isaac Sim Python experience found in {root}")
    launcher = AppLauncher(
        headless=True,
        enable_cameras=render,
        experience=str(experience),
        renderer="RayTracedLighting",
        anti_aliasing=0,
        rendering_mode="balanced",
    )
    from omni.kit.viewport.utility import get_active_viewport

    viewport = get_active_viewport()
    if viewport is not None:
        viewport.updates_enabled = False
    return launcher


def create_simulation():
    import carb
    import flatdict
    import isaaclab.sim.simulation_context as sim_module
    import toml
    from isaaclab.sim import SimulationCfg, SimulationContext

    preset_path = (
        Path(sim_module.__file__).resolve().parent / "../../../../apps/rendering_modes/balanced.kit"
    ).resolve()
    preset = flatdict.FlatDict(toml.load(preset_path), delimiter=".")
    paths = sorted({"/" + key.replace(".", "/") for key in preset} | {"/rtx/rendermode"})
    settings = carb.settings.get_settings()
    before = {path: settings.get(path) for path in paths}
    settings.set_string("/isaaclab/rendering/rendering_mode", "balanced")
    sim = SimulationContext(
        SimulationCfg(dt=1 / 120, render_interval=6, device="cpu", use_fabric=True, enable_scene_query_support=True)
    )
    after_lab = {path: settings.get(path) for path in paths}
    for path, value in before.items():
        if value is None:
            settings.destroy_item(path)
        else:
            settings.set(path, value)
    restored = {path: settings.get(path) for path in paths}
    if restored != before:
        raise RuntimeError("Failed to restore the Isaac Sim experience's renderer settings")
    settings.set_bool("/physics/disableContactProcessing", False)
    sim.fsop_render_settings = {
        "preset_path": str(preset_path),
        "base": before,
        "lab_preset": after_lab,
        "restored": restored,
    }
    return sim


def shutdown(app, drive=None):
    from isaaclab.sim import SimulationContext

    sim = SimulationContext.instance()
    if sim is not None:
        sim._disable_app_control_on_stop_handle = True
    if drive is not None:
        drive.close()
    elif sim is not None:
        sim.stop()
        sim.clear_instance()
    app.close(wait_for_replicator=False)
