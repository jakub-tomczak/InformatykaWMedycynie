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

def plot_graph(xAxis, yAxis, title='', xTitle=''):
    import matplotlib.pyplot as plt
    plt.plot(xAxis, yAxis)
    plt.ylabel(title)
    plt.xlabel(xTitle)
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


def fig2data(x, y):
    #plot_graph(x, y, '')
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig = plt.figure()
    plot = fig.add_subplot(111)
    plot.plot(y, x)
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    #buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    #buf = np.roll(buf, 3, axis=2)
    return buf

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


def clear_before_reconstruction(image):
    image_size = len(image)
    freqencies = fftfreq(image_size).reshape(-1, 1)
    omega = 2 * np.pi * freqencies
    fourier_filter = 2 * np.abs(freqencies)
    if use_cosine_in_fourier:
        fourier_filter *= np.cos(omega)
    projection = fft(image, axis=0) * fourier_filter
    return np.real(ifft(projection, axis=0))

def get_circle(size):
    circle = np.zeros((size, size))
    radius = size // 2
    center = (radius, radius)
    for y in range(0, size):
        for x in range(0, size):
            if (x-radius)**2 + (y-radius)**2 < radius**2:
                circle[y, x] = 1
    for i in np.arange(0,360, .1):
        x, y = calculatePositionSafe(i, size, center)
        circle[y, x] = 1
    return circle

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
            start, end = parallel_ray(start, angle, i*2, diameter, center)
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
            start, end = parallel_ray(start, angle, i*2, diameter, center)
        else:
            end = inclined_ray(angle, i*2, diameter, center)
        reconstruct_line(sinogram_value=sinogram[step, x], reconstruction_image=image, start_point=start, end_point=end, values=values)
        x+=1
def rootMeanSquareError2(input_image,output_image):
    error = 0
    n = 0
    error_sum = np.power(input_image - output_image, 2)
    suma = np.sum(error_sum)
    error = 1/(len(input_image)**2) * suma
    error = np.sqrt(error)
    return error


def rootMeanSquareError(input_image,output_image):
    error = 0
    n = 0
    for x in range(len(input_image)):
        for y in range(len(input_image[x])):
            pixelError = pow(input_image[x][y] - output_image[x][y],2)
            n+=1
            error += pixelError
    error = 1/n * error
    error = np.sqrt(error)
    return error

def darken(input_image):
    for x in range(len(input_image)):
        for y in range(len(input_image[x])):
            if input_image[x][y] <0.4:
                input_image[x][y] = 0
            else:
                input_image[x][y]-=0.4


def process(image, on_change, on_inverse_transform_change, on_finish):
    #new image
    new_image_size = get_new_image_shape(image)

    #keep original image matrix to get original values
    oryg_image = prepare_image(image)
    oryg_image = normalize_image_to_one(oryg_image)
    reconstructed = np.zeros(new_image_size)
    radon_steps = int(radon_angle // step_angle)
    sinogram_arr = np.zeros((radon_steps, n_detectors))

    circle_mask = get_circle(new_image_size[0])

    #create sinogram
    print('Creating sinogram')
    angle = 0
    for i in range(0, radon_steps):
        rays_image = np.zeros(new_image_size)
        display_status(i, radon_steps)
        radon(oryg_image, rays_image, angle, n_detectors, sinogram_arr, emission_angle, diameter=new_image_size[0], step=i)
        if on_change != None:
            on_change(oryg_image*255, rays_image, sinogram_arr*255, angle)
        angle += step_angle

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
    errors = np.zeros((radon_steps, 2))
    for i in range(0, radon_steps):
        display_status(i, radon_steps)
        inverse_radon(reconstructed, sinogram_arr, diameter=new_image_size[0], angle=angle, emission_angle=emission_angle,n_detectors=n_detectors, values = reconstruction_values, step=i)
        errors[i, 0] = angle
        errors[i, 1] = rootMeanSquareError2(oryg_image, normalize_image_to_one(reconstructed))
        if on_inverse_transform_change != None:
            on_inverse_transform_change(i+1, reconstructed*255)
        angle+=step_angle

    if use_convolution_in_output:
        reconstructed = convolve(reconstructed, kernel_reconstructed)
    #if use_gauss_in_reconstruction:
    #    reconstructed = gaussian(reconstructed)

    if normalize_output:
        reconstructed = normalize_image_to_one(reconstructed)

    darken(reconstructed)
    err = fig2data(errors[:, 1], errors[:, 0])
    if on_finish != None:
        on_finish(reconstructed*255, err)

    #plot_graph(errors[:, 0], errors[:, 1], 'Błąd średniokwadratowy', 'Numer iteracji')
    return sinogram_arr, reconstructed

#parameters
#   emiters detectors number
n_detectors = 50
#   angle between first and last ray
emission_angle = 80
#   rotation
radon_angle = 40
#   tomograph step
step_angle = 1

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
use_convolution_in_output = True
kernel_reconstructed = np.ones((9,9))

#file to transform
file = "watroba.png"
directory = os.path.join(os.getcwd(), "res")

filename_to_load = ''
image = None
generate_kernel()

def display_filename():
    print('file is ' + filename_to_load)

if __name__ == "__main__":
    print(file)
    image = misc.imread(os.path.join(directory, file), flatten=True).astype('float64')

    sinogram, reconstructed = process(image, None, None, None)
    plot_image(sinogram)
    plot_image(reconstructed)

