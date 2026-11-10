import subprocess
import sys

import pytest


@pytest.mark.parametrize("phase, mode, expected", [
    (False, "pp", "RUNTIME_REACHED"),
    (False, "op_tangent_pi_v1", "requires the pp learner controller"),
    (True, "op_tangent_pi_v1", "RUNTIME_REACHED"),
])
def test_collection_cli_controller_binding(phase, mode, expected):
    runner = '''
import runpy, sys, types
runtime = types.ModuleType("isaac_lab_fsop.runtime")
def launch(**kwargs):
    raise RuntimeError("RUNTIME_REACHED")
runtime.launch = launch
runtime.shutdown = lambda *args: None
sys.modules["isaac_lab_fsop.runtime"] = runtime
sys.argv = ["collect", *sys.argv[1:]]
runpy.run_module("isaac_lab_fsop.collect", run_name="__main__")
'''
    args = ["--case", "case.json", "--out", "capture", "--checkpoint", "checkpoint.pt",
            "--learner-controller", mode]
    if phase:
        args += ["--phase-schedule", "schedule.json"]
    result = subprocess.run([sys.executable, "-c", runner, *args], capture_output=True, text=True)
    assert result.returncode != 0
    assert expected in result.stderr
