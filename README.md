# Panorama Stitcher

A panorama stitching application built for COMP 493 (Computer Vision), Kathmandu University.
Feature matching with SIFT, a from-scratch homography (DLT) and RANSAC, warping and blending.

## Pipeline
SIFT keypoints -> ratio-test matching -> RANSAC homography -> warping -> blending

## Status
- [x] Feature matching
- [x] Homography from 4 point pairs (DLT, normalized)
- [x] RANSAC (matches OpenCV on test data)
- [ ] Warping and canvas
- [ ] Blending
- [ ] Multi-image stitching
- [ ] Interface and evaluation

## Setup
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

## Run
python -m pytest
python -m experiments.ransac_synthetic

## Results so far
Corner error (pixels) against the known true homography, by Lowe ratio threshold:

| Ratio | Least squares, all matches | Our RANSAC | OpenCV RANSAC |
|---|---|---|---|
| 0.6 | 22.72 | 0.08 | 0.08 |
| 0.7 | 681.71 | 0.08 | 0.08 |
| 0.8 | 989.76 | 0.08 | 0.09 |
| 0.9 | 860.10 | 0.07 | 0.05 |

## Credits
Test image: <photographer, source URL, license>
Team: Swarnim Bajracharya, Kshitiz Khanal