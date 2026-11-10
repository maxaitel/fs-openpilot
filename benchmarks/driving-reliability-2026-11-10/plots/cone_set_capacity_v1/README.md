# Capacity plot reproduction

`capacity.png` and `plot_receipt.json` preserve the original inline-rendered failed capacity screen. No per-update history was saved. `plot.py` renders the same receipt measurements using its checked-in template, into a new directory:

```sh
.venv/bin/python benchmarks/driving-reliability-2026-11-10/plots/cone_set_capacity_v1/plot.py --out runs/cone_set_capacity_plot_NEW
```

It refuses to overwrite an existing directory and records the input, source and output SHA256 values. Its presentation can differ from the original inline image. The limits apply to the two mean errors; individual anchor dots have no separate gate. These TRAIN anchors do not measure validation or driving reliability.
