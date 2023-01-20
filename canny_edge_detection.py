import math
from functools import partial, reduce

import cv2
import numpy as np
import skimage.filters as skif
from skimage.filters import median
from skimage.morphology import disk


def median_filter(image):
    img_median = median(image, disk(10), mode='constant', cval=0.0)

    return img_median


def calculate_gradient(image):
    horizontal_gradient = skif.sobel_h(image)
    vertical_gradient = skif.sobel_v(image)

    return horizontal_gradient, vertical_gradient


def calculate_polar_coordinates(vector):
    x, y = vector
    magnitude = (x ** 2 + y ** 2) ** (1/2)
    direction = np.arctan2(y, x)

    return magnitude, direction


def non_max_suppression(polar_coordinates):
    magnitude, direction = polar_coordinates

    slices = []
    cols, rows = magnitude.shape
    for dy in range(-1, 2):
        slices.append([])
        for dx in range(-1, 2):
            slices[-1].append(magnitude[1 + dy: cols - 1 +
                              dy, 1 + dx: rows - 1 + dx])

    # Create an empty output mask
    mask = np.zeros_like(magnitude, dtype=bool)

    # Discretize the direction to 8 angles
    direction = ((direction[1:-1, 1:-1] / (2 * math.pi) + 1/16) % 1) * 8 // 1

    directions = [(0, 1), (0, 0), (1, 0), (0, 2)]
    for i, (y, x) in enumerate(directions):
        is_pixel1_brighter = slices[1][1] > slices[y][x]
        is_pixel2_brighter = slices[1][1] > slices[2 - y][2 - x]
        is_retained = is_pixel1_brighter & is_pixel2_brighter
        is_oriented = (direction == i) | (direction == i + 4)
        mask[1:-1, 1:-1] |= is_oriented & is_retained

    return np.where(mask, magnitude, 0)


def apply_double_threshold(image):
    return np.digitize(image, bins=[0.025, 0.08])


def apply_hysteresis_threshold(image):
    image = skif.apply_hysteresis_threshold(image, 0.5, 1)
    image = np.where(image, 255, 0)
    return image


def compose(functions):
    return partial(reduce, lambda processed_value, func: func(processed_value), functions)


def detect_edges(image):
    processing_steps = compose([
        median_filter,
        calculate_gradient,
        calculate_polar_coordinates,
        non_max_suppression,
        apply_double_threshold,
        apply_hysteresis_threshold
    ])
    edge_image = processing_steps(image)
    return edge_image.astype(np.uint8)


def slice_image(img, *, left=0, right=0, top=0, bottom=0):
    right = right or -img.shape[0]
    bottom = bottom or -img.shape[1]
    return img[top:-bottom, left:-right]
