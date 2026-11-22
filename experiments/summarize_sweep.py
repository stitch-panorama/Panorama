import csv
import glob
import os
import sys

import numpy as np

pattern = sys.argv[1] if len(sys.argv) > 1 else "results/hpatches/sweep_*.csv"
print("file                      pairs  acc@1  acc@3  acc@5")
for path in sorted(glob.glob(pattern)):
    rows = [r for r in csv.DictReader(open(path)) if r["method"] == "ours"]
    e = np.array([float(r["error_px"]) for r in rows])
    print(f"{os.path.basename(path):<26}{len(e):5d} {np.mean(e <= 1):6.2f} "
          f"{np.mean(e <= 3):6.2f} {np.mean(e <= 5):6.2f}")