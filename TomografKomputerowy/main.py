from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
import os
from line_drawer import \
    bresenham, draw_line, calculatePosition, check_borders, calculatePositionSafe, reconstruct_line
import time
from scipy.ndimage.filters import convolve

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

def plot_graph(xAxis, yAxis, title):
    import matplotlib.pyplot as plt
    plt.plot(xAxis, yAxis)
    plt.ylabel(title)
    plt.show()


def inverse_radon(image, sinogram, diameter, angle, emission_angle, n_detectors):
    angle_between_rays = emission_angle / (n_detectors - 1)
    angles = np.arange(-emission_angle / 2, emission_angle / 2 + emission_angle / n_detectors, angle_between_rays)

    radius = diameter // 2 - 1
    center = (radius, radius)
    start = calculatePositionSafe(angle, diameter, center)

    x = 0
    for i in angles:
        if parallel_rays_mode:
            start, end = parallel_ray(start, angle, i*2, diameter, center)
        else:
            end = inclined_ray(angle, i*2, diameter, center)
        reconstruct_line(sinogram_value=sinogram[angle, x], reconstruction_image=image, start_point=start, end_point=end)
        x+=1

def normalize_image(image):
    maxV = np.max(image)
    if maxV == 0:
        return image
    return (image / maxV ) * 255

def parallel_ray(main_point_start, main_angle, minor_angle, diameter, center):
    start = calculatePositionSafe(main_angle + minor_angle, diameter, center)
    end = calculatePositionSafe(main_angle - 180 + (-1)*minor_angle, diameter, center)
    return start, end

def inclined_ray(main_angle, minor_angle, diameter, center):
    end_point = calculatePositionSafe(main_angle + minor_angle, diameter, center)
    return (diameter - end_point[0], diameter - end_point[1])


def radon(oryg_image, image, angle, n_detectors, sinogram_arr, emission_angle, diameter):
    angle_between_rays = emission_angle / (n_detectors - 1)
    angles = np.arange(-emission_angle/2, emission_angle/2 + emission_angle/n_detectors , angle_between_rays)

    radius = diameter // 2 - 1
    center = (radius, radius)
    start = calculatePositionSafe(angle, diameter, center)
    x = 0
    for i in angles:
        if parallel_rays_mode:
            start, end = parallel_ray(start, angle, i*2, diameter, center)
        else:
            end = inclined_ray(angle, i*2, diameter, center)
        line_sum = draw_line(oryg_image, image, start_point=start, end_point=end)
        sinogram_arr[angle, x] = line_sum
        x += 1


    if use_convolution_filter:
        sinogram_arr[angle, :] = convolve(sinogram_arr[angle, :], kernel)

    #plot_graph(range(0, sinogram_arr.shape[1]), line, "{a}".format(a =angle))
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
    #os.system('cls')
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
    rays_image = np.zeros(new_image_size)

    for i in range(0, radon_angle):
        rays_image = np.zeros(new_image_size)
        display_status(i, radon_angle)
        radon(oryg_image, rays_image, i, n_detectors, sinogram_arr, emission_angle, diameter=new_image_size[0])

    plot_image(sinogram_arr)
    sinogram_arr = normalize_image(sinogram_arr)

    plot_image(sinogram_arr)
    #return sinogram_arr, reconstructed

    print('Reconstructing image')
    #reconstruct image
    for i in range(0, radon_angle):
        display_status(i, radon_angle)
        inverse_radon(reconstructed, sinogram_arr, diameter=new_image_size[0], angle=i, emission_angle=emission_angle,n_detectors=n_detectors)
    return sinogram_arr, reconstructed

def debug(image):
     #im = discrete_radon_transform(image, 360)
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

def generate_kernel():
    top = -4/(np.pi**2)
    half = kernel_length // 2
    for i in range(0, kernel_length):
        kernel[i] = ( 0 if (i-half)%2==0 else (top/((i-half)**2)) )
    kernel[kernel_length // 2] = 1


radon_angle = 360
kernel_length = 100
kernel = np.zeros((kernel_length))
use_convolution_filter = True
parallel_rays_mode = True
if __name__ == "__main__":
    generate_kernel()
    file = "photo.png"
    directory = os.getcwd() + "\\res\\"
    #parametry
    #   liczba detektorów
    n_detectors = 200
    #   rozpiętość kątowa
    emission_angle = 30

    image = misc.imread('{dir}{file}'.format(dir=directory, file=file), flatten=True).astype('float64')
    #get image from debug source
    #image = debug(image)
    #expected_sinogram = discrete_radon_transform(image, radon_angle)
    #plot_image(expected_sinogram)

    sinogram, reconstructed = process(image)
    plot_image(sinogram)
    plot_image(reconstructed)

