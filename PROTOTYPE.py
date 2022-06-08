import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage.filters import median
from skimage.morphology import disk
import skimage.color as skic
import skimage.filters as skif
from skimage import io
from math import atan2, pi


def median_filter(image):
    img_median = median(image, disk(3), mode='constant', cval=0.0)
    cv2.imwrite('test.jpg', img_median)

    return img_median


def sobel(img):
    sobhimg = skif.sobel_h(img)
    sobvimg = skif.sobel_v(img)
    sobimg = (sobhimg ** 2 + sobvimg ** 2) ** (1/2)

    return sobimg


def gradient(img):
    sobhimg = skif.sobel_h(img)
    sobvimg = skif.sobel_v(img)

    return sobhimg, sobvimg


def polar(vector):
    x, y = vector
    magnitude = (x ** 2 + y ** 2) ** (1/2)
    direction = np.arctan2(y, x)

    return magnitude, direction


def non_max_suppression(gradient_magnitude, gradient_direction, verbose):
 
    image_row, image_col = gradient_magnitude.shape
 
    output = np.zeros(gradient_magnitude.shape)
 
    PI = 180
    
    for row in range(1, image_row - 1):
        for col in range(1, image_col - 1):
            direction = gradient_direction[row, col]
    
            if (0 <= direction < PI / 8) or (15 * PI / 8 <= direction <= 2 * PI):
                before_pixel = gradient_magnitude[row, col - 1]
                after_pixel = gradient_magnitude[row, col + 1]
    
            elif (PI / 8 <= direction < 3 * PI / 8) or (9 * PI / 8 <= direction < 11 * PI / 8):
                before_pixel = gradient_magnitude[row + 1, col - 1]
                after_pixel = gradient_magnitude[row - 1, col + 1]
    
            elif (3 * PI / 8 <= direction < 5 * PI / 8) or (11 * PI / 8 <= direction < 13 * PI / 8):
                before_pixel = gradient_magnitude[row - 1, col]
                after_pixel = gradient_magnitude[row + 1, col]
    
            else:
                before_pixel = gradient_magnitude[row - 1, col - 1]
                after_pixel = gradient_magnitude[row + 1, col + 1]
    
            if gradient_magnitude[row, col] >= before_pixel and gradient_magnitude[row, col] >= after_pixel:
                output[row, col] = gradient_magnitude[row, col]
 
    return output


def show(img):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(img, cmap=plt.cm.gray)
    ax.set_axis_off()
    plt.show()


if __name__== "__main__":
    file_path = "salt and pepper noise.png"
    image = cv2.imread(file_path, 0)
    img = median_filter(image)
    img = sobel(img)
    show(img)
    magnitude, direction = polar(gradient(img))
    img = non_max_suppression(magnitude, direction, "verbose")
    show(image)
    show(img)