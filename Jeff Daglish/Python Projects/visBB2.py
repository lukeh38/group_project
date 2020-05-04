import tkinter as tk
from tkinter import *
from tkinter import Tk, Canvas, Frame, filedialog, BOTH
from PIL import ImageTk, Image, ImageDraw 
import math
import os



def save_coords(event):
	#corners of box
	click_loc= [event.x,event.y]
	#save to file
	

#load next image
def next_img(imgs):
	img_label.img = ImageTk.PhotoImage(file=next(imgs))
	img_label.config(image=img_label.img)

root =tk.Tk()
img_dir=filedialog.askdirectory(parent=root, initialdir="E:/Python Projects/", title='Choose Folder')
os.chdir(img_dir)
imgs = iter(os.listdir(img_dir))

img_label = tk.Label(root)
img_label.pack()
img_label.bind("<Button-1>",save_coords)

btn = tk.Button(root, text='Next image', command=next_img)
btn.pack()
choosebtn = tk.Button(root, text ='Choose folder', command=chooseFile)
choosebtn.pack()

next_img()

root.mainloop()

f = open("labels.txt","w+")
for listitem in click_loc:
		f.write('%s '%listitem )
	f.write('\n')
f.close()