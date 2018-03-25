from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
import os

from skimage.filters import gaussian

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

def plot_graph(xAxis, yAxis, title=''):
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
        #return convolve(sinogram_arr[angle, :], kernel)
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

from numpy.fft import fftfreq, fft, ifft
def clear_before_reconstruction(image):
    #return gaussian(image)
    image_size = len(image)
    #size = 2**(int)(np.ceil( np.log2(image_size) ))
    freqencies = fftfreq(image_size).reshape(-1, 1)
    #pad_width = ((0,size - image_size), (0,0) ) #dodaj na szerokosci, nie na wysokosci
    #image = np.pad(image, pad_width, mode='constant', constant_values=0)
    omega = 2 * np.pi * freqencies
    fourier_filter = 2 * np.abs(freqencies)
    fourier_filter *= np.cos(omega)
    projection = fft(image, axis=0) * fourier_filter
    return np.real(ifft(projection, axis=0))


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

    #sinogram_arr = misc.imread('{dir}{file}'.format(dir='out\\', file='sinogram_800_80_head_withoutConvolution_with_fft.png'), flatten=True).astype('float64')

    sinogram_convolved = np.zeros((radon_angle, n_detectors))
    for i in range(0, radon_angle):
        rays_image = np.zeros(new_image_size)
        display_status(i, radon_angle)
        radon(oryg_image, rays_image, i, n_detectors, sinogram_arr, emission_angle, diameter=new_image_size[0])

    #misc.imsave("out\\sinogram_800_80_head_withoutConvolution_and_fft.png", sinogram_arr)
    #plot_image(sinogram_arr)
    #plot_image(sinogram_convolved)
    print('Reconstructing image')
    #reconstructed = misc.imread('{dir}{file}'.format(dir='out\\', file='reconstruction_200_30_head.png'), flatten=True).astype('float64')

    #sinogram_arr = normalize_image_to_one(sinogram_arr)
    #misc.imsave("out\\sinogram_800_80_head_withoutConvolution_with_fft.png", sinogram_arr)
    #fourier reconstruction - backprojection
    sinogram_arr = clear_before_reconstruction(sinogram_arr)

    #convolution
    for i in range(0, radon_angle):
        sinogram_arr[i, : ] = convolve(sinogram_arr[i, :], kernel)

    #sinogram_arr = normalize_image_to_one(sinogram_arr)
    #plot_image(sinogram_arr)

    #reconstruct image
    for i in range(0, radon_angle):
        display_status(i, radon_angle)
        inverse_radon(reconstructed, sinogram_arr, diameter=new_image_size[0], angle=i, emission_angle=emission_angle,n_detectors=n_detectors)

    reconstructed = normalize_image_to_one(reconstructed)
    counts = np.histogram(reconstructed, bins=np.arange(0, 1, 0.01))
    most = 0
    arg = 0
    for val, key in zip(counts[0], counts[1]):
        if val > most:
            most = val
            arg = key

    print(key)

    #MIN = np.percentile(reconstructed, 10)
    #MAX = np.percentile(reconstructed, 100-10)
    MIN = key - 0.01
    MAX = key + 0.01

    norm = reconstructed != key
    norm = reconstructed*norm
#    norm[norm > 1] = 1
 #   norm[norm < 0] = 0
    plot_image(norm)
    reconstructed = gaussian((reconstructed))


    '''
    reconstructed_c = convolve(reconstructed, np.ones((9,9)) )
    reconstructed_c2 = convolve(reconstructed, np.array([[0,1,0], [1,0,1], [0,1,0]]) )
    reconstructed_g = gaussian(reconstructed)
    misc.imsave("out\\reconstruction_with_fft_800_80_no_filter.png", reconstructed)
    misc.imsave("out\\reconstruction_with_fft_800_80_9_9_ones_kernel.png", reconstructed_c)
    misc.imsave("out\\reconstruction_with_fft_800_80_cross_kernel.png", reconstructed_c2)
    misc.imsave("out\\reconstruction_with_fft_800_80_gaussian.png", reconstructed_g)
    '''
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

#parametry
#   liczba detektorów
n_detectors = 50
#   rozpiętość kątowa
emission_angle = 50
radon_angle = 180
kernel_length = 100
file = "photo.png"
directory = os.getcwd() + "\\res\\"

kernel = np.zeros(( kernel_length))
use_convolution_filter = False

parallel_rays_mode = True
if __name__ == "__main__":
    generate_kernel()

    image = misc.imread('{dir}{file}'.format(dir=directory, file=file), flatten=True).astype('float64')
    #get image from debug source
    #image = debug(image)
    #expected_sinogram = discrete_radon_transform(image, radon_angle)
    #plot_image(expected_sinogram)

    sinogram, reconstructed = process(image)
    plot_image(sinogram)
    plot_image(reconstructed)

