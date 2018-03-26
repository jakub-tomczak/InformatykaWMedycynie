from enum import Enum

import numpy
import tkinter
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import Scale, Frame, Canvas, BOTH, filedialog, ttk
from tkinter import DoubleVar
from scipy import misc
import numpy as np
from random import random
import main


class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.grid(padx=4, pady=0)

        self.window_width = 1850
        self.window_height = 900
        self._geom='{x}x{y}+0+0'.format(x = self.window_width, y = self.window_height)
        self.master.geometry(self._geom)

        self.image_size = 450
        self.canvases = dict()
        self.scale_value = DoubleVar()

        self.create_window()
        self.create_widgets()


        #fullscreen
        #pad = 3
        #master.geometry("{0}x{1}+0+0".format(
        #    master.winfo_screenwidth()-pad, master.winfo_screenheight()-pad))
        #master.bind('<Escape>',self.toggle_geom)



    def create_widgets(self):
        self.create_buttons()
        #self.create_scroll_bar()
        self.create_canvases()
        self.create_parameters_box()
        self.create_progressbar()

        self.set_default_values()

    def create_buttons(self):
        #upload button
        self.upload_input_button = tk.Button(self, text="Wgraj obraz", command=self.upload_input_file,
                                             width=50, height=3, fg='white', bg='green')
        self.upload_input_button.grid(row=12, column=0, pady=15)
        #transform start button
        self.start_transform_button = tk.Button(self, text="Uruchom tomograf", command=start_radon_transform,
                                                width=50, height=3, fg='white', bg='green')
        self.start_transform_button.grid(row=13, column=0, pady=15)

    def create_window(self):
        self.frame = Frame(self.master, width=self.window_width, height=self.window_height)

    def upload_input_file(self):
        filename = filedialog.askopenfilename(filetypes=[('Image', 'jpg jpeg png gif')])
        if filename == "":
            return

        load_input_file(filename)
        img = Image.open(filename)
        self.set_image_on_canvas(img, ImageType.INPUT_IMAGE)

        self.reconstruction_progress['value'] = 0

    def set_image_on_canvas(self, img, image_type):
        canvas = self.canvases[image_type]
        if type(img) is np.ndarray:
            img = Image.frombytes('L', (img.shape[1],img.shape[0]), img.astype('b').tostring())
        img = img.resize((canvas.winfo_width(), canvas.winfo_height()), Image.ANTIALIAS)
        self.canvases[image_type].image = ImageTk.PhotoImage(img)
        self.canvases[image_type].create_image(0, 0, image=canvas.image, anchor=tk.NW)
        self.update()

    def create_canvases(self):
        x = 0
        for i in ImageType:
            self.canvases[i] = Canvas(self, width=self.image_size, height=self.image_size, bg='white')
            self.canvases[i].create_rectangle(2, 2, self.image_size, self.image_size)
            self.canvases[i].create_text(self.image_size // 2, self.image_size // 2, text=text_values[i])
            self.canvases[i].grid(row=0, column=x)
            x+=1

    def create_scroll_bar(self):
        self.scale = Scale(self, from_=0, to_=150, tickinterval=10,length=self.image_size - 20,
                           variable=self.scale_value, orient=tk.HORIZONTAL)
        self.scale.bind("<ButtonRelease-1>", method)
        self.scale.set(0)
        self.scale.grid(row=2, column=2)

    def create_parameters_box(self):
        default_font = ("Helvetica", 9)
        #error label
        self.error = tk.StringVar()
        tk.Label(self, textvariable=self.error, fg="red", font=("Helvetica", 16)).grid(row=2)

        tk.Label(self, text="Liczba emiterów", font=default_font).grid(row=3, sticky='w')
        self.detectors_number = tk.Entry(self, width=4, justify=tk.RIGHT)
        self.detectors_number.grid(row=3)

        tk.Label(self, text="Rozpiętość kątowa", font=default_font).grid(row=4, sticky='w')
        self.emission_angle = tk.Entry(self, width=4, justify=tk.RIGHT)
        self.emission_angle.grid(row=4)

        tk.Label(self, text="Obrót tomografu", font=default_font).grid(row=5, sticky='w')
        self.radon_angle = tk.Entry(self, width=4, justify=tk.RIGHT)
        self.radon_angle.grid(row=5)

        tk.Label(self, text="Krok", font=default_font).grid(row=6, sticky='w')
        self.step = tk.Entry(self, width=4, justify=tk.RIGHT)
        self.step.grid(row=6)

        self.sinogram_convolution = tkinter.IntVar(value=1)
        self.sinogram_convoltion_checkbox = tk.Checkbutton(self, text="Użyj splotu przy sinorgamie",
                                                     variable=self.sinogram_convolution ,
                                                     command=self.update_options)
        self.sinogram_convoltion_checkbox.grid(row=7, sticky='w')

        self.use_parallel_rays = tkinter.IntVar(value=1)
        self.parallel_rays_checkbox = tk.Checkbutton(self, text="Użyj promieni równoległych", variable=self.use_parallel_rays,
                                                          command=self.update_options)
        self.parallel_rays_checkbox.grid(row=8, sticky='w')


        self.use_fourier = tkinter.IntVar(value=1)
        self.fourier_checkbox = tk.Checkbutton(self, text="Użyj rekonstrukcji Fouriera",
                                                     variable=self.use_fourier,
                                                     command=self.update_options)
        self.fourier_checkbox.grid(row=9, sticky='w')

        self.use_convolution_at_end = tkinter.IntVar(value=1)
        self.convolution_end_checkbutton = tk.Checkbutton(self, text="Użyj splotu na wyjściu",
                                                          variable=self.use_convolution_at_end,
                                                          command=self.update_options)
        self.convolution_end_checkbutton.grid(row=10, sticky='w')

    def set_default_values(self):
        self.sinogram_convolution.set(1 if main.use_convolution_filter else 0)
        self.use_parallel_rays.set(1 if main.parallel_rays_mode else 0)
        self.use_convolution_at_end.set(1 if main.use_convolution_in_output else 0)
        self.use_fourier.set(1 if main.use_fourier_reconstruction else 0)

        self.detectors_number.insert(tk.END, main.n_detectors)
        self.radon_angle.insert(tk.END, main.radon_angle)
        self.emission_angle.insert(tk.END, main.emission_angle)
        self.step.insert(tk.END, main.step)

        self.reconstruction_progress['value'] = 0
        self.reconstruction_progress['maximum'] = main.radon_angle - 1

    def update_options(self):
        main.use_convolution_in_output = False if self.use_convolution_at_end.get() == 0 else True
        main.use_fourier_reconstruction = False if self.use_fourier.get() == 0 else True
        main.use_convolution_filter = False if self.sinogram_convolution.get() == 0 else True
        main.parallel_rays_mode = False if self.use_parallel_rays.get() == 0 else True

    def create_progressbar(self):
        self.reconstruction_progress = ttk.Progressbar(self, orient="horizontal",
                                                      length=self.image_size, mode="determinate")
        self.reconstruction_progress.grid(row=1, column=3, rowspan=2)


class ImageType(Enum):
    INPUT_IMAGE = 1
    SINOGRAM = 2
    ANIMATION_IMAGE = 3
    OUTPUT_IMAGE = 4

text_values = {
            ImageType.INPUT_IMAGE: 'Obraz wejściowy',
            ImageType.SINOGRAM: 'Sinogram',
            ImageType.ANIMATION_IMAGE: 'Animacja',
            ImageType.OUTPUT_IMAGE: 'Obraz wyjściowy'
}

def load_input_file(filename):
    main.filename_to_load = filename
    main.image = misc.imread(filename, flatten=True).astype('float64')
    app.error.set('')

def start_radon_transform():
    main.radon_angle = int(app.radon_angle.get())
    main.n_detectors = int(app.detectors_number.get())
    main.step = int(app.step.get())
    main.emission_angle = int(app.emission_angle.get())
    try:
        main.process(main.image, on_transform_change, on_inverse_transform_change, on_finish)
    except Exception as e:
        app.error.set(e)
    else:
        app.error.set('')
def on_transform_change(oryg, rays, sinogram, angle):
    app.set_image_on_canvas(sinogram, ImageType.SINOGRAM)
    app.set_image_on_canvas(rays+oryg, ImageType.ANIMATION_IMAGE)

def on_inverse_transform_change(angle):
    app.reconstruction_progress['value'] = angle
    app.reconstruction_progress.update()

def on_finish(img):
    app.set_image_on_canvas(img, ImageType.OUTPUT_IMAGE)

def method(event):
    print(app.scale_value.get())

if __name__ == "__main__":
    root = tk.Tk()
    app = Application(master=root)
    app.mainloop()
