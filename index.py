import cv2

import config
from canny_edge_detection import detect_edges, slice_image
from foot_measurement import convert_px_to_cm, get_foot_px


def process_image(file_path, bounds):
    img = cv2.imread(file_path, 0)
    img = slice_image(img, **bounds)
    img = detect_edges(img)
    return img


def get_foot_size(img):
    px = get_foot_px(img)
    cm = convert_px_to_cm(px)
    return cm


if __name__ == "__main__":
    file_path = "Samples/S02 24.5.jpg"
    img = process_image(file_path, config.image_bounds)
    cm = get_foot_size(img)

    print(cm)
