import cv2
from skimage.filters import median
from skimage.morphology import disk


def median_filter(image):
    img = cv2.imread(image, 0)
    img_median = median(img, disk(3), mode='constant', cval=0.0)
    cv2.imwrite('test.jpg', img_median)

    cv2.imshow("Original Image", img)
    cv2.imshow("Median Filtered Image", img_median)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__== "__main__":
    image = "salt and pepper noise.png"
    median_filter(image)