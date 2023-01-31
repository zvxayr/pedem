import os
import re

import cv2
import pandas as pd

import config
from canny_edge_detection import detect_edges, slice_image
from foot_measurement import get_foot_dimensions_px

samples_folder = 'Samples'
pattern = re.compile(r"(?P<id>\w+) (?P<cm>\d+\.\d+)")

data = []
for sample_file in os.listdir(samples_folder):
    match = pattern.search(sample_file)
    if not match:
        continue

    img = cv2.imread(f'{samples_folder}/{sample_file}', 0)
    img = slice_image(img, **config.image_bounds)
    img = detect_edges(img)
    px = get_foot_dimensions_px(img)

    data.append({"id": match.group("id"), "cm": match.group(
        "cm"), "h_px": px[0], "w_px": px[1]})

df = pd.DataFrame(data)
df.to_csv(config.output_csv, index=False)
