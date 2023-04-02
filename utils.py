import matplotlib.pyplot as plt


def show(image):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(image, cmap=plt.cm.gray)
    ax.set_axis_off()
    plt.show()
