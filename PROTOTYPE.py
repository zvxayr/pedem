from ast import Try
from logging import ERROR
from tkinter import END
import cv2
from cv2 import Sobel
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
import pickle
import functools
from dotenv import dotenv_values

config = dotenv_values('.env')


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
    pi = math.pi
    width, height = gradient_magnitude.shape

    gradient_direction = ((gradient_direction / (pi/8)) % 16)

    output = np.zeros(gradient_magnitude.shape)

    for i_y in range(height):
        for i_x in range(width):
            try:
                if (gradient_direction[i_x, i_y] >= 15) and (gradient_direction[i_x, i_y] < 1) or (gradient_direction[i_x, i_y] >= 7) and (gradient_direction[i_x, i_y] < 9):
                    pixel1 = gradient_magnitude[i_x + 1, i_y]
                    pixel2 = gradient_magnitude[i_x - 1, i_y]

                elif (gradient_direction[i_x, i_y] >= 1) and (gradient_direction[i_x, i_y] < 3) or (gradient_direction[i_x, i_y] >= 9) and (gradient_direction[i_x, i_y] < 11):
                    pixel1 = gradient_magnitude[i_x + 1, i_y + 1]
                    pixel2 = gradient_magnitude[i_x - 1, i_y - 1]

                elif (gradient_direction[i_x, i_y] >= 3) and (gradient_direction[i_x, i_y] < 5) or (gradient_direction[i_x, i_y] >= 11) and (gradient_direction[i_x, i_y] < 13):
                    pixel1 = gradient_magnitude[i_x, i_y + 1]
                    pixel2 = gradient_magnitude[i_x, i_y - 1]

                else:
                    pixel1 = gradient_magnitude[i_x - 1, i_y + 1]
                    pixel2 = gradient_magnitude[i_x + 1, i_y - 1]

                if (gradient_magnitude[i_x, i_y] > pixel1) and (gradient_magnitude[i_x, i_y] > pixel2):
                    output[i_x, i_y] = gradient_magnitude[i_x, i_y]

                else:
                    output[i_x, i_y] = 0

            except:
                pass

    return output

def use_model(filename):
    def with_reg(fun):
        with open(filename, 'rb') as f:
            reg = pickle.load(f)

        return functools.partial(fun, reg=reg)

    return with_reg

@use_model(config['model_file'])
def px2cm(x, reg):
    return reg.predict([[x]])[0]

def double_threshold(img):
    return np.digitize(img, bins=[0.04, 1])

def hysteresis_thresholding(img) :
    low_ratio = 0.01
    high_ratio = 0.1
    img = skimage.filters.apply_hysteresis_threshold(img, low_ratio, high_ratio)
    img = np.where(img, 255, 0)
    return img    

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
    img = hysteresis_thresholding(img)
    return img

def get_foot_px(img):
    contours, hierarchy = cv2.findContours(img, 1, 2)

    foot_px = 0
    for cnt in contours:
        (x, y), (w, h), a = rect = cv2.minAreaRect(cnt)
        foot_px = max(foot_px, w, h)
    
    return foot_px

if __name__== "__main__":
    # file_path = "TESTPIC.jpg"
    # img = cv2.imread(file_path, 0)
    # img = canny_edge(img).astype(np.uint8)

    img = np.empty((100, 100)).astype(np.uint8)
    img = cv2.circle(img, (50, 50), 25, (255, 255, 255), 1)
    px = get_foot_px(img)
    measurement = px2cm(px)

    print(measurement)
