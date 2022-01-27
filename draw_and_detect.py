from tkinter import *
from tkinter import simpledialog
from tkinter import messagebox
from PIL import Image
from PIL import ImageOps
from torchvision import transforms
import torchvision.datasets as datasets
from Model import Model
import torch
import csv
import matplotlib.pyplot as plt

to_bangla = {}

def image_loader(img_path):
    img = Image.open(img_path).convert('L') # convert to grayscale image
    loader = transforms.Compose([transforms.ToTensor()])

    img = loader(img).float()
    img = torch.tensor(img, requires_grad=True)
    img = img.unsqueeze(0)
    return img

def detect(img_path):
    model = Model().cuda()
    model.load_state_dict(torch.load("./data/model_bd_char_testing.dth"))   # load pre-trained model
    model.eval().cuda() # mode select

    img = image_loader(img_path)

    predict = model(img.cuda())
    predict = torch.nn.functional.softmax(predict, dim=1)

    predict = predict.squeeze(0)
    label = torch.max(predict.data, 0)[1]
    messagebox.showinfo('Info', 'Your Written Digit Is = ' + to_bangla[str(label.item())])

class Paint(object):

    DEFAULT_PEN_SIZE = 5.0
    DEFAULT_COLOR = 'black'

    def __init__(self):
        self.root = Tk()

        self.detect_button = Button(self.root, text='Detect', command=self.use_detect)
        self.detect_button.grid(row=0, column=0)

        self.pen_button = Button(self.root, text='Pen', command=self.use_pen)
        self.pen_button.grid(row=0, column=1)

        self.clear_button = Button(self.root, text='Clear', command=self.use_clear)
        self.clear_button.grid(row=0,column=2)

        self.eraser_button = Button(self.root, text='Eraser', command=self.use_eraser)
        self.eraser_button.grid(row=0, column=3)

        self.choose_size_button = Scale(self.root, from_=18, to=25, orient=HORIZONTAL)
        self.choose_size_button.grid(row=0, column=4)

        self.c = Canvas(self.root, bg='white', width=250, height=250)
        self.c.grid(row=1, columnspan=5)

        self.setup()
        self.root.mainloop()

    def setup(self):
        self.old_x = None
        self.old_y = None
        self.line_width = self.choose_size_button.get()
        self.color = self.DEFAULT_COLOR
        self.eraser_on = False
        self.active_button = self.pen_button
        self.c.bind('<B1-Motion>', self.paint)
        self.c.bind('<ButtonRelease-1>', self.reset)
        self.root.title("Draw")

    def use_detect(self):
        self.activate_button(self.detect_button)

    def use_pen(self):
        self.activate_button(self.pen_button)

    def use_clear(self):
        self.activate_button(self.clear_button, eraser_mode=False)

    def use_eraser(self):
        self.activate_button(self.eraser_button, eraser_mode=True)

    def activate_button(self, some_button, eraser_mode=False):
        self.active_button.config(relief=RAISED)
        some_button.config(relief=SUNKEN)
        self.active_button = some_button
        self.eraser_on = eraser_mode

        if some_button == self.clear_button:
            self.c.delete('all')
        elif some_button == self.detect_button:
            #filename = simpledialog.askstring("Filename", "Please Enter Filename")
            
            #if filename:
            file_path = './data/test'
            self.save_image(file_path)

            detect(file_path+'.png')


    def save_image(self, img_path):
        self.c.update()
        self.c.postscript(file=img_path+'.ps')   # saving postscript image

        img = Image.open(img_path+'.ps')
        # the model detects the white part of the image as digit. so, the 
        # image color needs to be inverted.
        img = ImageOps.invert(img)

        # resizing image to 28x28 pixel because the trained model can process
        # 28x28 size image only.
        img = img.resize((28,28), Image.ANTIALIAS)
        img.save(img_path+'.png', 'png')

    def paint(self, event):
        self.line_width = self.choose_size_button.get()
        paint_color = 'white' if self.eraser_on else self.color
        if self.old_x and self.old_y:
            self.c.create_line(self.old_x, self.old_y, event.x, event.y,
                               width=self.line_width, fill=paint_color,
                               capstyle=ROUND, smooth=TRUE, splinesteps=40)
        self.old_x = event.x
        self.old_y = event.y

    def reset(self, event):
        self.old_x, self.old_y = None, None


if __name__ == "__main__":

    # reading corresponding bangla characters of the labels
    with open('./data/ekushCSV/metaDataCSV.csv', encoding="UTF-8") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for rows in csv_reader:
            to_bangla[rows[0]] = rows[1]

    Paint()
