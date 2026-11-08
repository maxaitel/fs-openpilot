"""Wait for training, then evaluate, plot, and mark completion; errors stay visible."""

import json
import subprocess
import sys
import time
from pathlib import Path

OUT = Path("runs/spark_full")
while not (OUT / "pipeline.exit").exists():
    time.sleep(10)
if (OUT / "pipeline.exit").read_text().strip() != "0":
    raise RuntimeError("Training pipeline failed; inspect pipeline.log")
for label, script in [("evaluation", "scripts/evaluate_spark_full.py"), ("graphs", "scripts/plot_spark_full.py")]:
    with (OUT / f"{label}.log").open("w") as f:
        subprocess.run([sys.executable, script], stdout=f, stderr=subprocess.STDOUT, check=True)
(OUT / "finished.json").write_text(json.dumps({"finished_at": time.time(), "status": "complete"}))
