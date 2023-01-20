import os
import re

import cv2
import numpy as np

from canny_edge_detection import detect_edges, slice_image
from foot_measurement import convert_px_to_cm, get_foot_px

samples_folder = 'Samples'
pattern = re.compile(r"(?P<id>\w+) (?P<cm>\d+\.\d+)")

for sample_file in os.listdir(samples_folder):
    match = pattern.search(sample_file)
    if not match:
        continue

    img = cv2.imread(f'{samples_folder}/{sample_file}', 0)
    img = slice_image(img, left=100, right=100, top=20, bottom=20)
    img = detect_edges(img)
    px = get_foot_px(img)
    cm = convert_px_to_cm(px)

    print(match.group("cm"), cm)
