PROFILE = {
    "renderer": "RayTracedLighting",
    "anti_aliasing": 0,
    "dlss_exec_mode": 0,
    "dl_denoiser": True,
    "motion_blur": False,
    "subframes": 16,
    "reason": "4 and 8 subframes retained stale marker ghosts in HDR lighting; 16 passed moving-camera projection.",
}


def configure_rendering():
    import carb

    settings = carb.settings.get_settings()
    settings.set("/rtx-transient/dldenoiser/enabled", PROFILE["dl_denoiser"])
    settings.set("/rtx/post/motionblur/enabled", PROFILE["motion_blur"])
    settings.set("/rtx/post/dlss/execMode", PROFILE["dlss_exec_mode"])


def capture(world):
    import omni.replicator.core as rep

    world.render()
    rep.orchestrator.step(rt_subframes=PROFILE["subframes"], delta_time=0.0, pause_timeline=False)
