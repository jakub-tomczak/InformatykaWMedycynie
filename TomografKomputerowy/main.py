from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
import os
from line_drawer import bresenham, calculate_line, calculatePosition, check_borders

file = "photo.png"
directory = os.getcwd() + "\\res\\"

image = misc.imread('{dir}{file}'.format(dir = directory, file = file), flatten=True).astype('float64')

def radon_example():
    # Read image as 64bit float gray scale
    radon = discrete_radon_transform(image, len(image))


    # Plot the original and the radon transformed image
    plt.subplot(1, 3, 1), plt.imshow(image, cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.subplot(1, 3, 2), plt.imshow(radon, cmap='gray')
    plt.xticks([]), plt.yticks([])


size = 200
radi = (int)(size / 2 - 1)
image2 = np.zeros((size, size))


def draw_line_example():
    for i in range(0, 360):
        x1, y1 = calculatePosition(i, radi, (radi, radi))
        x1, y1 = check_borders((x1, y1), size)

        x2 = size - x1
        y2 = size - y1
        # print('data is', 'start',x1, y1, 'end',x2, y2)
        bresenham(image2, (x1, y1), (x2, y2))

    plt.subplot(1, 1, 1), plt.imshow(image2, cmap='gray')
    plt.show()

def discrete_radon_transform(image, steps):
    R = np.ones((len(image), steps), dtype='float64')
    #rotation = misc.imrotate(image, -45 * 180 / steps).astype('float64')
    #R[:, 45] = sum(rotation)
    for s in range(30,55):
        rotation = misc.imrotate(image, -s*180/steps).astype('float64')
        R[:,s] = sum(rotation)
    return R


def draw_emitter(image, angle):
    for i in range(-8, 9, 4):
        calculate_line(image, angle + i)

image = np.zeros((200, 200))
draw_emitter(image, 20)
plt.subplot(1, 1, 1), plt.imshow(image, cmap='gray')
plt.show()
