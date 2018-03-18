from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
import os
from line_drawer import \
    bresenham, draw_line, calculatePosition, check_borders, calculatePositionSafe

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


def draw_emitter(image, angle, n_detectors):
    if n_detectors % 2 == 0:
        n_detectors += 1
    #Draw main line
    big_diameter = len(image)
    big_radius = (int)(big_diameter / 2) - 1
    big_center = (big_radius, big_radius)
    start = calculatePositionSafe(angle, big_diameter, big_center)
    draw_line(image, angle=angle, diameter=big_diameter, center=big_center)

    x2 = big_diameter - start[0]
    y2 = big_diameter - start[1]

    small_diameter = (int)(big_diameter/2)
    small_center = ( (int)((x2+big_center[0])/2), (int)((y2+big_center[1])/2) )
    small_radius = (int)(small_diameter/2)-1

    angle_between_detectors = 5
    deviation = angle_between_detectors * (int)(n_detectors / 2)
    for i in range(angle - deviation, angle + deviation + 1, angle_between_detectors):
        if i == angle:
            continue
        end = calculatePosition(i, small_radius, small_center )
        end = (2*small_center[0] - end[0], 2*small_center[1] - end[1])
        end = check_borders(end, big_diameter)
        print(start, end)
        draw_line(image, start_point=start, end_point=end)

image = np.zeros((500, 500))
n_detectors = 5
emitter_angle = 140
draw_emitter(image, emitter_angle, n_detectors)
plt.subplot(1, 1, 1), plt.imshow(image, cmap='gray')
plt.show()
