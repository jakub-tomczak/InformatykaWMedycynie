from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
import os

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

def bresenham(image, start, end):
    x1,y1 = start
    x2,y2 = end

    #pomijamy zal 1
    #if x2 < x1 and y2 > y1:

    #zal 2 - kat miedzy styczna a osia X < 45 stopni
    #if x2 - x1 < y2 - y1:
    #    raise Exception("Alg bresenhama: niespełnione drugie założenie! x1:{x1} y1:{y1} x2:{x2} y2:{y2}".format(x1 = x1, x2=x2, y1=y1, y2=y2));
    xi = 1
    yi = 1
    if x2 - x1 < 0:
        xi = -1
    if y2 - y1 < 0:
        y1 = -1

    dx = np.abs(x2 - x1)
    dy = np.abs(y2 - y1)
    e = dx / 2
    image[y1,x1] = 1

    if dx == 0 or dy == 0:
        draw_straight_line(image, start, end)
    else:
        if dx > dy:
            #os wiodaca OX
            for i in range(0, dx):
                x1 = x1 + xi
                e = e - dy
                if e < 0:
                    y1 = y1 + yi
                    e = e + dx
                #print(x1, y1)
                image[y1, x1] = 1
        else:
            #os wiodaca OY
            for i in range(0, dy):
                y1 = y1 + yi
                e = e - dx
                if e < 0:
                    x1 = x1 + xi
                    e = e + dy
                #print(x1, y1)
                image[y1, x1] = 1

def draw_straight_line(image, start, end):
    if start[0] == end[0]:
        image[start[0],:] = 1
        print("all",start[0], "y:", start[1], end[1])
    if end[1] == start[1]:
        image[:,start[1]] = 1
        print("all", start[1], "x:", start[0], end[0])

#R[:,2] => wszystkie wiersze 2 kolumna
#R[a,b] -> a-ty wiersz, b-ta kolumna
def calculatePosition(angle, radius, circle_middle = (0,0)):
    rad = angle / 180 * np.pi
    return (circle_middle[0] + (int)(radius*np.cos(rad)), circle_middle[1] - (int)(radius*np.sin(rad)))

def drawCircle(image, angle, radius, circle_middle):
    (x, y) = calculatePosition(angle, radius, circle_middle)
    image[x, y] = .5
    #plt.subplot(1, 1, 1), plt.imshow(image, cmap='gray')
    #plt.show()


def check_value(val, size):
    if val <= 0:
        val = 1
    if val >=size:
        val = size -1
    return val
def check_borders(point, size):
    return (check_value(point[0], size),check_value(point[1], size))

def draw_emitter(image, angle):
    for i in range(-8, 9, 4):
        calculate_line(image, angle + i)

def calculate_line(image, angle):
    size = len(image)
    radius = (int)(size / 2) - 1
    center = (radius, radius)
    x1, y1 = calculatePosition(angle, radius, center)
    x1, y1 = check_borders((x1,y1), size)

    x2 = size - x1
    y2 = size - y1
    # print('data is', 'start',x1, y1, 'end',x2, y2)
    bresenham(image, (x1, y1), (x2, y2))


image = np.zeros((200, 200))
draw_emitter(image, 20)
plt.subplot(1, 1, 1), plt.imshow(image, cmap='gray')
plt.show()
