import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage.filters import median
from skimage.morphology import disk
import skimage.color as skic
import skimage.filters as skif
from skimage import io
from math import atan2, pi, tau
import math


def median_filter(image):
    img_median = median(image, disk(3), mode='constant', cval=0.0)
    return img_median


def gradient(img):
    sobhimg = skif.sobel_h(img)
    sobvimg = skif.sobel_v(img)

    return sobhimg, sobvimg


def polar(vector):
    x, y = vector
    magnitude = (x ** 2 + y ** 2) ** (1/2)
    direction = np.arctan2(y, x)

    return magnitude, direction

def non_max_suppression(gradient_magnitude, gradient_direction):
    image_row, image_col = gradient_magnitude.shape
    output = np.zeros(gradient_magnitude.shape)

    direction = gradient_direction[1:-1, 1:-1]
    direction = ((direction / tau + 1/16) % 1) * 8 // 1
    
    cc = gradient_magnitude[1:image_row-1, 1:image_col-1]
    ll = (gradient_magnitude[1:image_row-1, 0:image_col-2] < cc).astype(int)
    rr = (gradient_magnitude[1:image_row-1, 2:image_col-0] < cc).astype(int)
    uu = (gradient_magnitude[0:image_row-2, 1:image_col-1] < cc).astype(int)
    bb = (gradient_magnitude[2:image_row-0, 1:image_col-1] < cc).astype(int)
    ul = (gradient_magnitude[0:image_row-2, 0:image_col-2] < cc).astype(int)
    ur = (gradient_magnitude[0:image_row-2, 2:image_col-0] < cc).astype(int)
    bl = (gradient_magnitude[2:image_row-0, 0:image_col-2] < cc).astype(int)
    br = (gradient_magnitude[2:image_row-0, 2:image_col-0] < cc).astype(int)

    output[1:-1, 1:-1] = (
        np.logical_or(direction == 0, direction == 4) * (ll * rr) +
        np.logical_or(direction == 1, direction == 5) * (bl * ur) +
        np.logical_or(direction == 2, direction == 6) * (uu * bb) +
        np.logical_or(direction == 3, direction == 7) * (ul * br)
    )

    return output * gradient_magnitude


def show(img):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(img, cmap=plt.cm.gray)
    ax.set_axis_off()
    plt.show()

def canny_edge(image):
    img = median_filter(image)
    magnitude, direction = polar(gradient(img))
    img = non_max_suppression(magnitude, direction)
    img = double_threshold(img)
    
    return img

if __name__== "__main__":
    file_path = "TESTPIC.jpg"
    image = cv2.imread(file_path, 0)
    new_image = canny_edge(image)
    show(image)
    show(new_image)