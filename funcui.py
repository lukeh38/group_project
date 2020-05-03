# A interface demonstrating the functionality of calling models from a central location


# Imports for UI
from tkinter import *
import camera as cam
import threading
from PIL import Image, ImageTk


# Variable that references the Yolo_Model class from camera.py
cameramain=cam.Yolo_Model()

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
    print("Currently prototype: Future release to remove images")

# A function that can be called to close the window and exit the program
def close_window():
    exit(0)
    my_window.destroy()

# A class (guiFrame) containing components of the user interface
class guiFrame(Frame):
    def __init__(self, my_window):
        Frame.__init__(self, my_window)

        # A string variable that can be used to alter the display text for the "current model" label within the UI
        modelVar = StringVar()
        modelVar.set('Yolo Model')

        # A button that calls the "runTraining" function upon the event of a click.
        self.button1 = Button(self, text="Run Training", command= lambda:  [modelVar.set('Yolo Model'), runTraining()], width=13, height=2, bd=0, bg="white", font="calibri")
        self.button1.grid(row=0, column=1, padx=10, pady=10)
        # A button that calls the "runGestures" function upon the event of a click.
        self.button2 = Button(self, text="Detect Gestures", command=lambda: runGestures(), width=13, height=2, bd=0, bg="white", font="calibri")
        self.button2.grid(row=0, column=2, padx=10, pady=10)
        # A button that calls the "runFiles" function upon the event of a click.
        self.button3 = Button(self, text="Clear images", command=lambda: runFiles(), width=13, height=2, bd=0, bg="white", font="calibri")
        self.button3.grid(row=0, column=3, padx=10, pady=10)
        # A button that calls the "close_window" function upon the event of a click.
        self.exitbtn = Button(self, text="Exit", command = lambda :close_window(), width=13, height=2, bd=0, bg="white", font="calibri")
        self.exitbtn.grid(row=0, column=4, padx=10, pady=10)
        # A title label for the canvas which displays the video capture
        self.canvastitle = Label(my_window, text = "Video Capture")
        self.canvastitle.grid(row = 1, column = 0)
        # A title for the most recent capture image
        self.imagetitle = Label(my_window, text = "Most Recent Capture")
        self.imagetitle.grid(row = 1, column = 1)
        # A canvas for displaying the openCV video feed upon the "runTraining" or "runGestures" functions being called
        self.vidwindow = Canvas(my_window, height= 400, width = 450, relief = RIDGE)
        self.vidwindow.grid(row=2, column=0, pady = 3, padx = 3)
        # A label that will be used in future development to display accuracy of a model
        self.accuracytitle = Label(my_window, text = "Model Accuracy: ", anchor = W)
        self.accuracytitle.grid(row = 3, column = 0)
        # A label that displays the current model being used
        self.modeltitle = Label(my_window, text = "Current model being used: " , anchor = W)
        self.modeltitle.grid(row = 3, column = 1)
        self.modelResult=Label(my_window, textvariable=modelVar)
        self.modelResult.grid(row=3, column=2)
        # Importing and resizing the image created in camera.py of a captured gesture, and using PhotoImage to load make it a usable variable in tkinter
        imagefile = Image.open("img.jpg")
        imagefile = imagefile.resize((100,100), Image.ANTIALIAS)
        photo = ImageTk.PhotoImage(imagefile)
        # A label containing the image stored in the "photo" variable, retrieved from camera.py
        self.imagelabel = Label(image=photo, width = 100, height = 100, anchor = N)
        self.imagelabel.image = photo
        self.imagelabel.place(x=540, y=95)

# Construction of UI window, including size, state, colour and creation of a frame for the UI
my_window = Tk()
my_window.state('normal')
my_window.title("Gesture Recognition")
my_window.geometry('700x520')
my_window.config(bg="#33ccff")
frameA = guiFrame(my_window)
frameA.grid(row=0, column=0)
frameA.config(bg="#33ccff")

# Mainloop for UI

my_window.mainloop()