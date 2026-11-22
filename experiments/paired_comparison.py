import csv
import sys
from math import comb


def load(path):
    return {(r["sequence"], r["pair"]): float(r["error_px"])
            for r in csv.DictReader(open(path)) if r["method"] == "ours"}


a, b = load(sys.argv[1]), load(sys.argv[2])
thr = float(sys.argv[3]) if len(sys.argv) > 3 else 3.0
only_a = sum(1 for k in a if a[k] <= thr < b[k])
only_b = sum(1 for k in a if b[k] <= thr < a[k])
n = only_a + only_b
k = min(only_a, only_b)
p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
print(f"within {thr:g} px: only first run {only_a}, only second run {only_b}, p = {p:.3f}")