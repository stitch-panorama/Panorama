import csv
import sys

import matplotlib.pyplot as plt
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "results/hpatches_baseline.csv"
rows = list(csv.DictReader(open(path)))
thresholds = np.linspace(0, 10, 101)
names = [("ours", "ours (RANSAC)"), ("opencv", "OpenCV RANSAC"), ("lstsq", "least squares")]

fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
for ax, kind in zip(axes, ["i", "v", "all"]):
    for m, label in names:
        e = np.array([float(r["error_px"]) for r in rows
                      if r["method"] == m and (kind == "all" or r["kind"] == kind)])
        ax.plot(thresholds, [(e <= t).mean() for t in thresholds], label=label)
    ax.set_title({"i": "illumination", "v": "viewpoint", "all": "all scenes"}[kind])
    ax.set_xlabel("corner error threshold (px)")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("fraction of image pairs")
axes[0].legend()
fig.tight_layout()
fig.savefig("results/hpatches_accuracy_curves.png", dpi=150)
print("saved results/hpatches_accuracy_curves.png")