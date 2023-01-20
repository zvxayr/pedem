import cv2

import config
from canny_edge_detection import detect_edges, slice_image
from foot_measurement import convert_px_to_cm, get_foot_px

if __name__ == "__main__":
    file_path = "Samples/S02 24.5.jpg"
    img = cv2.imread(file_path, 0)
    img = slice_image(img, **config.image_bounds)
    img = detect_edges(img)
    px = get_foot_px(img)
    cm = px2cm(px)

    print(cm)
