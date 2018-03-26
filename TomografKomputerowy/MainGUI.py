from enum import Enum

import numpy
import tkinter
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import Scale, Frame, Canvas, BOTH, filedialog
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
        self.window_height = 700
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
        self.create_scroll_bar()
        self.create_canvases()

    def create_buttons(self):
        #upload button
        self.upload_input_button = tk.Button(self, text="Wgraj obraz", command=self.upload_input_file)
        self.upload_input_button.grid(row=1, column=0, pady=2)
        #transform start button
        self.start_transform_button = tk.Button(self, text="Uruchom tomograf", command=start_radon_transform)
        self.start_transform_button.grid(row=2, column=0, sticky=tk.W)

    def create_window(self):
        self.frame = Frame(self.master, width=self.window_width, height=self.window_height)

    def upload_input_file(self):
        filename = filedialog.askopenfilename(filetypes=[('Image', 'jpg jpeg png gif')])
        if filename == "":
            pass

        load_input_file(filename)
        img = Image.open(filename)
        self.set_image_on_canvas(img, ImageType.INPUT_IMAGE)

    def set_image_on_canvas(self, img, image_type):
        canvas = self.canvases[image_type]
        if type(img) is np.ndarray:
            img = Image.frombytes('L', (img.shape[1],img.shape[0]), img.astype('b').tostring())
        img = img.resize((canvas.winfo_width(), canvas.winfo_height()), Image.ANTIALIAS)
        self.canvases[image_type].image = ImageTk.PhotoImage(img)
        canvas.create_image(0, 0, image=canvas.image, anchor=tk.NW)
        root.update()

    def create_canvases(self):
        x = 0
        for i in ImageType:
            self.canvases[i] = Canvas(self, width=self.image_size, height=self.image_size, bg='white')
            self.canvases[i].create_rectangle(2, 2, self.image_size, self.image_size)
            self.canvases[i].create_text(self.image_size // 2, self.image_size // 2, text=text_values[i])
            self.canvases[i].grid(row=0, column=x)
            x+=1

    def change_command_buttons_state(self, state):
        self.upload_input_button['state'] = 'normal' if state else 'disable'
        self.start_transform_button['state'] = 'normal' if state else 'disable'

    def create_scroll_bar(self):
        self.scale = Scale(self, from_=0, to_=150, tickinterval=10,length=self.image_size - 20,
                           variable=self.scale_value, orient=tk.HORIZONTAL)
        self.scale.bind("<ButtonRelease-1>", method)
        self.scale.set(0)
        self.scale.grid(row=2, column=1)



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

def start_radon_transform():
    app.change_command_buttons_state(False)
    try:
        main.process(main.image, on_change)
    except:
        pass
    app.change_command_buttons_state(False)

def on_change(oryg, rays, sinogram, angle):
    app.set_image_on_canvas(sinogram, ImageType.SINOGRAM)
    app.set_image_on_canvas(rays+oryg, ImageType.ANIMATION_IMAGE)

def method(event):
    print(app.scale_value.get())

if __name__ == "__main__":
    root = tk.Tk()
    app = Application(master=root)
    app.mainloop()