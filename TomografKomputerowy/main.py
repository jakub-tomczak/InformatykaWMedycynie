from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
import os
from line_drawer import \
    bresenham, draw_line, calculatePosition, check_borders, calculatePositionSafe, reconstruct_line


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

def inverse_radon(image, sinogram, diameter, angle, emission_angle, n_detectors):
    angle_between_rays = emission_angle / (n_detectors - 1)
    angles = np.arange(-emission_angle / 2, emission_angle / 2 + emission_angle / n_detectors, angle_between_rays)

    radius = diameter // 2 - 1
    center = (radius, radius)
    start_point = calculatePositionSafe(angle, diameter, center)

    x = 0
    for y in angles:
        end_point = calculatePositionSafe(angle + y*2, diameter, center)
        end_point = (diameter - end_point[0], diameter - end_point[1])
        reconstruct_line(start_point=start_point, end_point=end_point, sinogram_value=sinogram[angle, x], reconstruction_image=image)
        x+=1
def normalize_image(image):
    maxV = np.max(image)
    return (image / maxV ) * 255

def radon(oryg_image, image, angle, n_detectors, sinogram_arr, emission_angle, diameter):
    angle_between_rays = emission_angle / (n_detectors - 1)
    angles = np.arange(-emission_angle/2, emission_angle/2 + emission_angle/n_detectors , angle_between_rays)

    radius = diameter // 2 - 1
    center = (radius, radius)
    start_point = calculatePositionSafe(angle, diameter, center)

    x = 0
    for i in angles:
        end_point = calculatePositionSafe(angle + i*2, diameter, center)
        end_point = (diameter - end_point[0], diameter - end_point[1])
        sum = draw_line(oryg_image, image, start_point=start_point, end_point=end_point)
        sinogram_arr[angle, x] = sum
        x += 1

def get_new_image_shape(old_image):
    if old_image.shape[0] != old_image.shape[1]:
        raise Exception('Image is not a square!')
    #size is sqrt(2)*a -> diagonal of a square
    size = (int)(np.sqrt(2) * max(old_image.shape[0], old_image.shape[1]) + 10)
    return (size, size)


#creates quare matrix that is sqrt(2)*size(image)+10 long
def prepare_image(image):
    size = get_new_image_shape(image)
    new_image = np.zeros(size)
    dy = image.shape[0] // 2
    dx = image.shape[1] // 2
    new_center = size[0] // 2
    # put image into new_image
    new_image[new_center - dy: new_center + dy, new_center - dx: new_center + dx] = image

    return new_image



def display_status(num, all):
    os.system('cls')
    p = num * 100 // all
    print("{status}>{spaces}{percent}%".format(status='-' * (p), spaces=(100 - p - 1) * ' ', percent=p))

def process(image):
    #new image
    new_image_size = get_new_image_shape(image)

    #keep original image matrix to get original values
    oryg_image = prepare_image(image)
    reconstructed = np.zeros(new_image_size)
    sinogram_arr = np.zeros((radon_angle, n_detectors))

    #create sinogram
    print("Creating sinogram")
    for i in range(0, 360):
        rays_image = np.zeros(new_image_size)
        display_status(i, radon_angle)
        radon(oryg_image, rays_image, i, n_detectors, sinogram_arr, emission_angle, diameter=new_image_size[0])
    sinogram_arr = normalize_image(sinogram_arr)


    print('Reconstructing image')
    #reconstruct image
    for i in range(0, radon_angle):
        display_status(i, radon_angle)
        inverse_radon(reconstructed, sinogram_arr, diameter=new_image_size[0], angle=i, emission_angle=emission_angle,n_detectors=n_detectors)

    return sinogram_arr, reconstructed

def debug(image):
    # im = discrete_radon_transform(image, 360)
    # im = misc.imrotate(im, 90).astype('float64')

    # plot_image(im)

     angle = 360
     im = np.zeros(image.shape)
     l = len(image) // 4
     im[l:2*l, l:2*l] = 255
     im[l*2:l*3, l*2:l*3] = 255
     return im
     sinogram_arr = np.zeros((angle, len(image)))
     radon(im, im, 0, 60, None, 20, len(image))
     for i in range(0,angle):
        rotation = misc.imrotate(im, i).astype('float64')
        result = np.multiply(rotation, image)
        #plot_image(result)
        #plot_image(rotation)
        res = sum(result)
        sinogram_arr[i:] = res


radon_angle = 360
if __name__ == "__main__":
    file = "photo.png"
    directory = os.getcwd() + "\\res\\"
    #parametry
    #   liczba detektorów
    n_detectors = 100
    #   rozpiętość kątowa
    emission_angle = 120

    image = misc.imread('{dir}{file}'.format(dir=directory, file=file), flatten=True).astype('float64')
    #get image from debug source
    #image = debug(image)
    sinogram, reconstructed = process(image)
    plot_image(reconstructed)
    expected_sinogram = discrete_radon_transform(image, radon_angle)
    plot_image(expected_sinogram)

