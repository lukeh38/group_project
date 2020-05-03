# A secondary UI that has less functionality, but aesthetically fulfils the intended design

# Imports
from tkinter import *
from PIL import ImageTk, Image
import cv2
import threading
import camera as cam


# Variable that references the Yolo_Model class from camera.py
cameramain=cam.Yolo_Model()
filename="img.jpg"


# A function that calls the model_trainer function from camera.py and starts a thread
def runTraining():
    cameramain.model_trainer()
    th = threading.Thread(target=runTraining())
    th.setDaemon(True)
    th.start()

# A function that calls the model_detector function from camera.py and starts a thread
def runGestures():
    cameramain.model_detector()
    th = threading.Thread(target=runGestures())
    th.setDaemon(True)
    th.start()
# A function that calls the removeFiles function from camera.py and starts a thread
def runFiles():
    cameramain.removeFiles()
    th = threading.Thread(target=runFiles())
    th.setDaemon(True)
    th.start()

# A function that can be called to close the window and exit the program
def close_window():
    exit(0)
    root.destroy()

root = Tk()
# Create a frame
root.state('normal')
root.title("Gesture Recognition")

root.config(bg="#33ccff")
app = Frame(root, bg="#33ccff")
app.grid()

root.rowconfigure(0,weight=1, uniform='row')

# A button that calls the "runTraining" function upon the event of a click.
button1 = Button(app,  text="Run Training", width=13, height=2, bd=0, bg="white", font="calibri")
button1.grid(row=0, column=1, padx=5, pady=10)
# A button that calls the "runGestures" function upon the event of a click.
button2 = Button(app, text="Detect Gestures", width=13, height=2, bd=0, bg="white", font="calibri")
button2.grid(row=0, column=2, padx=5, pady=10)
# A button that calls the "runFiles" function upon the event of a click.
button3 = Button(app, text="Clear images",width=13, height=2, bd=0, bg="white", font="calibri")
button3.grid(row=0, column=3, padx=5, pady=10)
# A button that calls the "close_window" function upon the event of a click.
exitbtn = Button(app, text="Exit", command = lambda: close_window(), width=13, height=2, bd=0, bg="white", font="calibri")
exitbtn.grid(row=0, column=4, pady=10)

# A title for the most recent capture image
imagetitle = Label(app, text = "Most Recent Capture", anchor = N)
imagetitle.grid(row = 0, column = 5)

# A label that will be used in future development to display accuracy of a model
accuracytitle = Label(app, text = "Model Accuracy: ", anchor = W)
accuracytitle.grid(row = 2, column = 1)

# Importing and resizing the image created in camera.py of a captured gesture, and using PhotoImage to load make it a usable variable in tkinter
imagefile = Image.open("img.jpg")
imagefile = imagefile.resize((75, 75), Image.ANTIALIAS)
photo = ImageTk.PhotoImage(imagefile)
# A label containing the image stored in the "photo" variable, retrieved from camera.py
imagelabel = Label(image=photo, width=75, height=75, anchor=N)
imagelabel.image = photo
imagelabel.place(x=670, y=75)

# Create a label in the frame
lmain = Label(app)
lmain.grid(row=1, column =1, columnspan=4)

# Capture from camera
cap = cv2.VideoCapture(0)

# function for video streaming
def video_stream():
    ret, frame = cap.read()
    if ret == True:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)
        lmain.after(1, video_stream)

video_stream()
root.mainloop()