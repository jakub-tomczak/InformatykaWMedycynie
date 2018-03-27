from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
import os
from numpy.fft import fftfreq, fft, ifft

#from skimage.filters import gaussian

from line_drawer import \
    bresenham, draw_line, calculatePosition, check_borders, calculatePositionSafe, reconstruct_line
import time
from scipy.ndimage.filters import convolve

def plot_image(image):
    plt.subplot(1, 1, 1), plt.imshow(image, cmap='gray')
    plt.show()

def plot_graph(xAxis, yAxis, title=''):
    import matplotlib.pyplot as plt
    plt.plot(xAxis, yAxis)
    plt.ylabel(title)
    plt.show()

def generate_kernel():
    top = -4/(np.pi**2)
    half = kernel_length // 2
    for i in range(0, kernel_length):
        kernel[i] = ( 0 if (i-half)%2==0 else (top/((i-half)**2)) )
    kernel[kernel_length // 2] = 1

def normalize_image_to_one(image):
    min = np.min(image)
    if min < 0:
        #make all values positive
        image += np.abs(min)
    distance = np.max(image)
    return (image / distance)

def normalize_image(image):
    maxV = np.max(image)
    if maxV == 0:
        return image
    return (image / maxV ) * 255

def parallel_ray( main_angle, minor_angle, diameter, center):
    start = calculatePositionSafe(main_angle + minor_angle, diameter, center)
    end = calculatePositionSafe(main_angle - 180 + (-1)*minor_angle, diameter, center)
    return start, end

def inclined_ray(main_angle, minor_angle, diameter, center):
    end_point = calculatePositionSafe(main_angle + minor_angle, diameter, center)
    return (diameter - end_point[0], diameter - end_point[1])

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

def check_parameters_values():
    if type(image) is type(None):
        raise Exception("Niepoprawne zdjęcie")
    if n_detectors < 1 or n_detectors >= len(image):
        raise Exception("Niepoprawna liczba emiterów")
    if emission_angle<1 or emission_angle >= 360:
        raise Exception("Niepoprawny kąt rozwarcia")
    if radon_angle <= 0 or radon_angle>360:
        raise Exception("Niepoprawny kąt obrotu tomografu")
    if step_angle <= 0 or step_angle >= radon_angle:
        raise Exception("Niepoprawny krok przesunięcia")


def clear_before_reconstruction(image):
    image_size = len(image)
    freqencies = fftfreq(image_size).reshape(-1, 1)
    omega = 2 * np.pi * freqencies
    fourier_filter = 2 * np.abs(freqencies)
#    if use_cosine_in_fourier:
#        fourier_filter *= np.cos(omega)
    projection = fft(image, axis=0) * fourier_filter
    return np.real(ifft(projection, axis=0))


def display_status(num, all):
    #os.system('cls')
    p = num * 100 // all
    print("{status}>{spaces}{percent}%".format(status='-' * (p), spaces=(100 - p - 1) * ' ', percent=p))


def radon(oryg_image, image, angle, n_detectors, sinogram_arr, emission_angle, diameter, step):
    angle_between_rays = emission_angle / (n_detectors - 1)
    angles = np.arange(-emission_angle/2, emission_angle/2 + emission_angle/n_detectors , angle_between_rays)

    radius = diameter // 2 - 1
    center = (radius, radius)
    start = calculatePositionSafe(angle, diameter, center)
    x = 0
    for i in angles:
        if parallel_rays_mode:
            start, end = parallel_ray( angle, i*2, diameter, center )
        else:
            end = inclined_ray(angle, i*2, diameter, center)
        line_sum = draw_line(oryg_image, image, start_point=start, end_point=end)
        sinogram_arr[step, x] = line_sum
        x += 1

def inverse_radon(image, sinogram, diameter, angle, emission_angle, n_detectors, values, step):
    angle_between_rays = emission_angle / (n_detectors - 1)
    angles = np.arange(-emission_angle / 2, emission_angle / 2 + emission_angle / n_detectors, angle_between_rays)

    radius = diameter // 2 - 1
    center = (radius, radius)
    start = calculatePositionSafe(angle, diameter, center)

    x = 0
    for i in angles:
        if parallel_rays_mode:
            start, end = parallel_ray( angle, i*2, diameter, center )
        else:
            end = inclined_ray(angle, i*2, diameter, center)
        reconstruct_line(sinogram_value=sinogram[step, x], reconstruction_image=image, start_point=start, end_point=end, values=values)
        x+=1


def process(image, on_change, on_inverse_transform_change, on_finish):
    check_parameters_values()
    #new image
    new_image_size = get_new_image_shape(image)

    #keep original image matrix to get original values
    oryg_image = prepare_image(image)
    reconstructed = np.zeros(new_image_size)
    radon_steps = int(radon_angle // step_angle)
    sinogram_arr = np.zeros((radon_steps, n_detectors))

    #create sinogram
    print('Creating sinogram')
    angle = 0
    for i in range(0, radon_steps):
        rays_image = np.zeros(new_image_size)
        display_status(i, radon_steps)
        radon(oryg_image, rays_image, angle, n_detectors, sinogram_arr, emission_angle, diameter=new_image_size[0], step=i)
        if on_change != None:
            on_change(oryg_image, rays_image, sinogram_arr, angle)
        angle += step_angle
        #plot_image(rays_image+oryg_image)

    print('Reconstructing image')

    #fourier reconstruction - backprojection
    if use_fourier_reconstruction:
        sinogram_arr = clear_before_reconstruction(sinogram_arr)

    #convolution
    if use_convolution_filter:
        for i in range(0, radon_steps):
            sinogram_arr[i, : ] = convolve(sinogram_arr[i, :], kernel)

    #reconstruct image
    reconstruction_values = np.ones(new_image_size)
    angle = 0
    for i in range(0, radon_steps):
        display_status(i, radon_steps)
        inverse_radon(reconstructed, sinogram_arr, diameter=new_image_size[0], angle=angle, emission_angle=emission_angle,n_detectors=n_detectors, values = reconstruction_values, step=i)
        if on_inverse_transform_change != None:
            on_inverse_transform_change(i+1)
        angle+=step_angle
    if use_convolution_in_output:
        reconstructed = convolve(reconstructed, kernel_reconstructed)
    #if use_gauss_in_reconstruction:
    #    reconstructed = gaussian(reconstructed)

    if normalize_output:
        reconstructed = normalize_image_to_one(reconstructed)

    if on_finish != None:
        on_finish(reconstructed*255)

    return sinogram_arr, reconstructed

#parameters
#   emiters detectors number
n_detectors = 10
#   angle between first and last ray
emission_angle = 10
#   rotation
radon_angle = 180
#   tomograph step
step_angle = 20.3

parallel_rays_mode = False
normalize_output = True

#filters parameters
#sinogram convolution
kernel_length = 5
kernel = np.zeros(( kernel_length))
use_convolution_filter = True

#fourier reconstruction
use_fourier_reconstruction = False
use_cosine_in_fourier = True

#output convolution
use_gauss_in_reconstruction = True
use_convolution_in_output = False
kernel_reconstructed = np.ones((9,9))

#file to transform
file = "photo.png"
directory = os.getcwd() + "\\res\\"

filename_to_load = ''
image = None
generate_kernel()

def display_filename():
    print('file is ' + filename_to_load)

if __name__ == "__main__":
    image = misc.imread('{dir}{file}'.format(dir=directory, file=file), flatten=True).astype('float64')

    sinogram, reconstructed = process(image, None, None, None)
    plot_image(sinogram)
    plot_image(reconstructed)

