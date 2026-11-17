import cv2
from stitcher.pipeline import stitch

pairs = {
    "synthetic": ("data/synthetic/c.jpg", "data/synthetic/d.jpg"),
    "cabinet": ("data/cabinet/a.jpg", "data/cabinet/b.jpg"),
}
for name, (pa, pb) in pairs.items():
    res = stitch([cv2.imread(pa), cv2.imread(pb)])
    print(name, res.ok, res.message, res.stats)
    if res.ok:
        cv2.imwrite(f"results/pipeline_{name}.jpg", res.panorama)