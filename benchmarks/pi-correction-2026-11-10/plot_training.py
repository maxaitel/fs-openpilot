import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


root = Path(__file__).resolve().parent
history = [json.loads(line) for line in (root / 'training/history.jsonl').read_text().splitlines()]
selection = json.loads((root / 'training/selection_receipt.json').read_text())
assert [row['step'] for row in history] == list(range(1, 501))
validation = [(0, selection['candidates'][0]['validation']['aggregate'])]
validation += [(row['step'], row['validation']['aggregate']) for row in history if 'validation' in row]
limit = next(gate['limit'] for gate in selection['candidates'][-1]['gates'] if gate['metric'] == 'aggregate.steer')
out = root.parents[1] / 'runs/skidpad_correction_report'
out.mkdir(parents=True, exist_ok=True)
fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout='constrained')
for ax, key, title in zip(axes, ['native_relative_acceleration', 'plan_l1', 'steer'],
                         ['Acceleration consistency', 'Plan L1', 'Validation steering loss'], strict=True):
    if key != 'steer':
        ax.plot([row['step'] for row in history], [row[key] for row in history], alpha=.35, label='TRAIN batch')
    ax.plot([step for step, row in validation], [row[key] for step, row in validation], 'o-', label='Validation')
    if key == 'steer':
        ax.axhline(limit, color='black', linestyle='--', label='Frozen ceiling')
    ax.set(title=title, xlabel='Optimizer update')
    ax.grid(alpha=.2)
    ax.legend()
fig.suptitle('500 updates completed. No trained checkpoint qualified.')
fig.savefig(out / 'training-consistency.png', dpi=150)
plt.close(fig)
paths = [root / 'training/history.jsonl', root / 'training/selection_receipt.json', Path(__file__)]
receipt = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
(out / 'training-consistency.inputs.json').write_text(json.dumps(receipt, indent=2) + '\n')
