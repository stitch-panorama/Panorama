import os

import cv2
import gradio as gr

from stitcher.pipeline import stitch


def run(files, ratio, thresh):
    if not files or len(files) < 2:
        return None, "Please upload at least two overlapping photos."
    # uploaded files get temporary folder names, so sort by the file's own name
    paths = sorted((getattr(f, "name", f) for f in files), key=os.path.basename)
    images = [cv2.imread(p) for p in paths]
    if any(im is None for im in images):
        return None, "One of the files could not be read as an image."

    result = stitch(images, ratio=ratio, thresh=thresh)
    if not result.ok:
        return None, "Could not stitch these images: " + result.message

    st = result.stats
    lines = [f"Stitched {len(images)} images in {st['runtime_s']} s "
         f"({st['projection']} projection, reference image {st['reference'] + 1}, "
         f"canvas {st['canvas'][0]} x {st['canvas'][1]} px)."]
    for p in st["pairs"]:
        lines.append(f"Images {p['pair'][0]} and {p['pair'][1]}: "
                     f"{p['inliers']} inliers out of {p['matches']} matches")
    return cv2.cvtColor(result.panorama, cv2.COLOR_BGR2RGB), "\n".join(lines)


demo = gr.Interface(
    fn=run,
    inputs=[
        gr.File(label="Overlapping photos (name them 1.jpg, 2.jpg, ... left to right)",
                file_count="multiple", file_types=["image"]),
        gr.Slider(0.5, 0.95, value=0.7, step=0.05, label="Ratio-test threshold"),
        gr.Slider(1, 8, value=3, step=0.5, label="RANSAC inlier threshold (pixels)"),
    ],
    outputs=[gr.Image(label="Panorama"), gr.Textbox(label="Details")],
    title="Panorama Stitcher",
    description="SIFT features, RANSAC homography (written from scratch), warping and blending. "
                "Photos are processed on the server for this demo. Please don't upload private photos.",
)

if __name__ == "__main__":
    demo.launch()