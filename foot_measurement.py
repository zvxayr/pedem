import pickle
from functools import lru_cache

from cv2 import dilate, erode, findContours, minAreaRect
from skimage.morphology import disk

import config


def process_edge_image(edge_image, kernel):
    edge_image = dilate(edge_image, kernel, iterations=1)
    edge_image = erode(edge_image, kernel, iterations=1)
    return edge_image


def find_largest_closed_boundary_side_px(edge_image):
    contours, _ = findContours(edge_image, 1, 2)

    largest_rect = (0, 0)
    for contour in contours:
        (_, _), (w, h), _ = minAreaRect(contour)
        largest_rect = max(largest_rect, tuple(sorted((w, h), reverse=True)))

    return largest_rect


def get_foot_dimensions_px(edge_image):
    processed_edge_image = process_edge_image(edge_image, disk(4))
    foot_dimensions_px = find_largest_closed_boundary_side_px(
        processed_edge_image)
    return foot_dimensions_px


@lru_cache
def load_model(model_source):
    with open(model_source, 'rb') as f:
        return pickle.load(f)


def convert_px_to_cm(px, model_source=config.model_file):
    reg = load_model(model_source)
    return reg.predict([[px]])[0]
