import pickle
from functools import lru_cache

from cv2 import dilate, erode, findContours, minAreaRect
from dotenv import dotenv_values
from skimage.morphology import disk

import config


def process_edge_image(edge_img, kernel):
    edge_img = dilate(edge_img, kernel, iterations=1)
    edge_img = erode(edge_img, kernel, iterations=1)
    return edge_img


def find_largest_closed_boundary_side_px(edge_img):
    contours, _ = findContours(edge_img, 1, 2)

    largest = 0
    for cnt in contours:
        (_, _), (w, h), _ = minAreaRect(cnt)
        largest = max(largest, w, h)

    return largest


def get_foot_px(edge_img):
    processed_edge_img = process_edge_image(edge_img, disk(4))
    foot_px = find_largest_closed_boundary_side_px(processed_edge_img)
    return foot_px


@lru_cache
def load_model(model_source):
    with open(model_source, 'rb') as f:
        return pickle.load(f)


def convert_px_to_cm(x, model_source=config.model_file):
    reg = load_model(model_source)
    return reg.predict([[x]])[0]
