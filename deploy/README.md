---
title: Panorama Stitcher
emoji: 🏞️
colorFrom: indigo
colorTo: yellow
sdk: gradio
sdk_version: 6.32.0
python_version: "3.11"
app_file: app.py
pinned: false
---

# Panorama Stitcher

SIFT features, RANSAC homography, warping and blending, with cylindrical projection for wide scenes.
Homography, RANSAC, warping and blending are implemented from scratch; SIFT and the matcher come from OpenCV.
Upload 2 or more overlapping photos, named so they sort left to right.

Course project, COMP 493, Kathmandu University. Code: https://github.com/stitch-panorama/Panorama