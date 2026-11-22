import csv
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "results/hpatches/sweep_sift_r0.8.csv"
good_px = 5.0
rows = [r for r in csv.DictReader(open(path)) if r["method"] == "ours"]


def accepted(r):
    return int(r["inliers"]) > 8 + 0.3 * int(r["matches"])


tp = fp = tn = fn = 0
for r in rows:
    good = float(r["error_px"]) <= good_px
    acc = accepted(r)
    tp += acc and good
    fp += acc and not good
    tn += (not acc) and (not good)
    fn += (not acc) and good

n = len(rows)
print(f"pairs {n}")
print(f"accepted and good (<= {good_px:g} px): {tp}")
print(f"accepted but bad (false accept):    {fp}")
print(f"rejected and bad (correct reject):  {tn}")
print(f"rejected but good (false reject):   {fn}")
print(f"acc@{good_px:g} overall:           {(tp + fn) / n:.2f}")
print(f"accuracy among accepted:     {tp / max(tp + fp, 1):.2f}  (accepted {tp + fp} of {n})")
print(f"bad pairs caught:            {tn / max(tn + fp, 1):.2f}")
print(f"good pairs wrongly rejected: {fn / max(tp + fn, 1):.2f}")
big = [r for r in rows if float(r["error_px"]) >= 100]
print(f"catastrophic (>= 100 px): {len(big)}, rejected by the rule: {sum(not accepted(r) for r in big)}")