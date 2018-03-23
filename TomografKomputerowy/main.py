from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
import os
from line_drawer import \
    bresenham, draw_line, calculatePosition, check_borders, calculatePositionSafe


def radon_example(image):
    # Read image as 64bit float gray scale
    radon = discrete_radon_transform(image, len(image))

    # Plot the original and the radon transformed image
    plt.subplot(1,2, 1)
    plt.imshow(image, cmap='gray')
    plt.subplot(1,2,2)
    plt.imshow(radon)
    plt.show()

def discrete_radon_transform(image, steps):
    R = np.zeros((len(image), steps), dtype='float64')
    #rotation = misc.imrotate(image, -45 * 180 / steps).astype('float64')
    #R[:, 45] = sum(rotation)
    for s in range(0, steps):
        rotation = misc.imrotate(image, -s*180/steps).astype('float64')
        #plot_image(rotation)
        R[:,s] = sum(rotation)
    return R

def plot_image(image):
    plt.subplot(1, 1, 1), plt.imshow(image, cmap='gray')
    plt.show()

def draw_rays_old(oryg_image, img, angle, n_detectors, sinogram_arr, emission_angle, diameter):
    if n_detectors % 2 == 0:
        n_detectors += 1
    #Draw main line
    if diameter == 0:
        diameter = len(image)
    big_diameter = diameter
    big_radius = (int)(big_diameter / 2) - 1
    big_center = (big_radius, big_radius)
    start = calculatePositionSafe(angle, big_diameter, big_center)
    draw_line(oryg_image, image, sinogram_arr, angle=angle, diameter=big_diameter, center=big_center)

    x2 = big_diameter - start[0]
    y2 = big_diameter - start[1]

    small_diameter = (int)(big_diameter/2)
    #get center of smaller circle
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
        draw_line(oryg_image, image, sinogram_arr, start_point=start, end_point=end)

def calculate_emission_ray(start_point, diameter, angle):
    end_point = (diameter - start_point[0], diameter - start_point[1])
    end_point = check_borders(end_point, diameter)
    z = np.tan(angle*np.pi/180)*diameter
    dx = np.sin(angle*np.pi/180)*z
    dy = np.cos(angle*np.pi/180)*z
    if angle < -24 or angle > 24:
        print(angle, dx, dy, end_point)
    x = (int)(end_point[0] + dx)
    y = (int)(end_point[1] + dy)
    return check_borders((x, y), diameter)

def draw_rays_new(oryg_image, image, angle, n_detectors, sinogram_arr, emission_angle, diameter):
    angle_between_rays = emission_angle / (n_detectors - 1)
    angles = np.arange(-emission_angle/2, emission_angle/2 + emission_angle/n_detectors , angle_between_rays)

    radius = diameter // 2 - 1
    center = (radius, radius)
    main_ray_coordinates = calculatePositionSafe(angle, diameter, center)

    x = 0
    for i in angles:
        start_point = main_ray_coordinates
        end_point = calculatePositionSafe(angle + i*2, diameter, center)
        end_point = (diameter - end_point[0], diameter - end_point[1])
        sum = draw_line(oryg_image, image, image, sinogram_arr, start_point=start_point, end_point=end_point)
        #print(i,x, sum)
        #sinogram_arr[angle, x] = sum
        x += 1

def draw_emitter(oryg_image,image, angle, n_detectors, sinogram_arr, emission_angle, diameter = 0, mode = 1):
    if mode == -1:
        draw_rays_old(oryg_image, image, angle, n_detectors, sinogram_arr, emission_angle, diameter)
    else:
        draw_rays_new(oryg_image, image, angle, n_detectors, sinogram_arr, emission_angle, diameter)


#creates quare matrix that is sqrt(2)*size(image)+10 long
def prepare_image(image):
    if image.shape[0] != image.shape[1]:
        raise Exception('Image is not a square!')
    #size is sqrt(2)*a -> diagonal of a square
    size = (int)(np.sqrt(2) * max(image.shape[0], image.shape[1]) + 10)
    new_image = np.zeros((size, size))
    dy = image.shape[0] // 2
    dx = image.shape[1] // 2
    new_center = size // 2
    # put image into new_image
    new_image[new_center - dy: new_center + dy, new_center - dx: new_center + dx] = image

    return new_image

def process(image):
    new_image = prepare_image(image)
    #keep original image matrix to get original values
    oryg_image = np.array(new_image)
    sinogram_arr = np.zeros((180, n_detectors))

    #loop that draws rays
    #for i in range(80, 100):
    #    draw_emitter(oryg_image, new_image, i, n_detectors, sinogram_arr, emission_angle, diameter=len(new_image))
    draw_emitter(oryg_image, new_image, 90, n_detectors, sinogram_arr, emission_angle, diameter=len(new_image))
    plot_image(new_image)

def debug(image):
     angle = 360
     im = np.zeros(image.shape)
     sinogram_arr = np.zeros((angle, len(image)))
     draw_rays_new(im, im, 0, 60, None, 20, len(image))
     for i in range(0,angle):
        rotation = misc.imrotate(im, i ).astype('float64')
        result = np.multiply(rotation, image)
        #plot_image(result)
        #plot_image(rotation)
        res = sum(result)
        sinogram_arr[i:] = res
     plot_image(sinogram_arr)


if __name__ == "__main__":
    file = "photo.png"
    directory = os.getcwd() + "\\res\\"
    #parametry
    #   liczba detektorów
    n_detectors = 100
    #   rozpiętość kątowa
    emission_angle = 50

    image = misc.imread('{dir}{file}'.format(dir=directory, file=file), flatten=True).astype('float64')

    #debug(image)

    process(image)
    # plt.subplot(1, 1, 1), plt.imshow(new_image, cmap='gray')
    #plt.show()

