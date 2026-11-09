from dataclasses import dataclass
from pathlib import Path

from fsop.sim.env import FSEnv, SimCfg
from fsop.sim.randomization import dr_preset
from fsop.track.fixed_layouts import skidpad
from fsop.track.generate import TrackGenCfg, generate_track
from fsop.track.io import load_json
from isaac_eval.parity_cases import _InitializationRenderer, build_environment, get_case


@dataclass(frozen=True)
class Case:
    id: str
    split: str
    discipline: str
    seed: int
    dr: str = "heavy"
    variant: str = "visual_stress_v1"
    track_seed: int | None = None
    frozen_case: str | None = None
    lateral_start: float = 0.0
    yaw_start: float = 0.0


def environment(case: Case, tracks=Path("data/native_tracks")):
    if case.frozen_case:
        return build_environment(get_case(case.frozen_case))
    if case.discipline == "skidpad":
        track = skidpad()
    else:
        if case.track_seed is None:
            raise ValueError("Trackdrive requires a track seed")
        tracks.mkdir(parents=True, exist_ok=True)
        path = tracks / f"trackdrive_{case.track_seed:06d}.json"
        if path.exists():
            track = load_json(path)
        else:
            from fsop.track.io import save_json

            track = generate_track(case.track_seed, TrackGenCfg())
            save_json(track, path)
        track.laps = 1
    cfg = SimCfg(dr=case.dr, max_time=120, venue_prob=1.0)
    cfg.noise.enabled = False
    dr = dr_preset(case.dr)
    dr.panorama_dir = "artifacts/panoramas"
    env = FSEnv(track, cfg, seed=case.seed, renderer=_InitializationRenderer(), dr_cfg=dr)
    env.renderer = None
    return env
