import numpy as np
import os
from numpy import expand_dims
import cv2
from keras_predict import pred
from trainer import predicter
from keras.preprocessing.image import img_to_array
import time
from im_to_txt import m
from random import random 
import shutil
class Yolo_Model:
    # Sets the intial parameters used within the definition of the class.  
    def __init__(self):
        # Enables the video capture
        self.cap = cv2.VideoCapture(0)
        # Sets the temporary directory used for the captured frame
        self.filename = "img.jpg"
        # Sets the destination of the training directory
        self.train_dir = "new_training"
        try:
            os.mkdir(self.train_dir)
        except:
            print("directory exists")
        
    def func(self,curr):
        boo = ""
        image  = cv2.imread(self.filename)
        os.remove(self.filename)
        print("Enter 'T' if the label is correct or 'F' is the label is incorrect!")
        boo = input(curr[0]+"\n")
        try:
            if boo == 'T':
                name = str(random())+ '.jpg'
                name = os.path.join(self.train_dir,name)
                try:
                    cv2.imwrite(name,image)
                except:
                    print("Could not save the image to " + str(name))
                m(name, curr)
            image  = cv2
        except ValueError as e:
            print(e)
            print("Value not recognised")

    def model_detector(self):
        while(self.cap.isOpened()):
            ret, frame = self.cap.read()
            if ret==True:
                # time.sleep(1)
                frame = cv2.flip(frame,+1)
                cv2.imshow('window',frame)
                try:
                    cv2.imwrite(self.filename,frame)
                except:
                    print("Could not save the image to img.jpg")
                output = predicter(0.6, self.filename)

                os.remove(self.filename)
                # os.remove(photo_filename)
                while (len(output) >= 5): 
                    curr = output[0:6]
                    output = output[6:]
                    label = curr[0]
                    try:
                        image = cv2.rectangle(frame,(curr[1],curr[4]),(curr[3],curr[2]),(0,0,255),3)
                        image = cv2.putText(frame,label,(curr[1],(curr[2]-10)),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
                    except:
                        print("Could not annote the image. Please ensure the co-ordinates are correct")    
                    cv2.imshow('window',image)

            # Used to detstroy the image capture window
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            else:
                break    
    # This function is used to train the system. It detects a gesture being shown by the user and applies
    # a label. It then prompts the user, asking if the label identified was correct. If yes, the yolo format
    # text file is created, needed for training, and the image is saved into the training directory. The user
    # is also shown the detected image with the label and bounding box. 
    def model_trainer(self):
        # Constant loop for capturing frames for the webcam. 
        while(True):
            # Captures the image and frame number.
            ret, frame = self.cap.read()
            if ret==True:
                # Additional sleep timer can be used for waiting in between capturing frames. Turned
                # off for real time detection. 
                # time.sleep(1)
                # Flips the frame so it appears the right way for the user.
                frame = cv2.flip(frame,+1)
                # Shows the user the frame that has been captured by the camera. This is constantly 
                # updated each time a new frame is captured.
                cv2.imshow('window',frame)
                # Try and except block used incase the system cannot write the captured frame. The frame
                # is saved so it can be used by the prediction file.
                try:
                    cv2.imwrite(self.filename,frame)
                except:
                    print("Could not save the image to img.jpg")
                # The predict method from trainer.py is called. This will return the prediction of the captured
                # image along with the the parameters needed for the boudnding box and prediction score. The parameter
                # 0.6 corrosponds to threshold of how confident the model is about the label applied. Any lower and the 
                # prediction will be discounted. 0.6 is lower than normal to allow for additonal classifcation to be made
                # during training.
                output = predicter(0.6,self.filename)
                # If the length of the array returned from the model prediction is greater than 5,i.e. a prediction was made
                # then the first 6 values, corrosponding to the length of one prediction, will be upacked until there are no more
                # predictions from that frame. 
                while (len(output) >= 5):
                    # The first 6 values corrosponding to the first prediction are extracted 
                    curr = output[0:6]
                    output = output[6:]
                    # The predicted label is the first value within the returned array
                    label = curr[0]
                    # A try block is used incase the given parameters for the bounding box are incorrect or cannot be labeled
                    # within the image. 
                    try:
                        image = cv2.rectangle(frame,(curr[1],curr[4]),(curr[3],curr[2]),(0,0,255),3)
                        image = cv2.putText(frame,label,(curr[1],(curr[2]-10)),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
                    except:
                        print("Could not annote the image. Please ensure the co-ordinates are correct")    
                    cv2.imshow('window',image) 
                    # This tunction is called to check if the predicted label was correct and if so use the image for training.
                    self.func(curr)                                                                   
            # Used to destroy the image capture window
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            else:
                break
# Adds additional ethics and security in removing the training directory
# from the system.
    def removeFiles(self):
        # Try and except used incase the training directory could not be located.
        try:
            # Removes the training directory.
            shutil.rmtree(self.train_dir, ignore_errors=True)
        # Throws an OS error as the file cannot be found, printing user instructions on the error.
        except OSError as e:
            print("Error: ")
            print(e)
# The main menu for the user to select the options for the system. This menu calls the functions
# used for training, detecting, and removing stored images.
    def main(self):
        menu = input("Press 1 to train the model: \nWarning training will store images of you!!\nPress 2 to detect the hand gesture: \nPress 3 to remove any images stored during training\n")
        if menu == '1':
            while(True):
                self.model_trainer()
        elif menu == '2':
            while(True):
                self.model_detector()
        elif menu == '3':
            self.removeFiles()
# The final else statement catches the program should a unexpected value be entered.
        else:
            print("Error. Correct option not selected")

# Automatically calls the main method within the Yolo_Model class
if __name__ == "__main__":
    model = Yolo_Model()
    # Call the main method 
    model.main()
    print("Closing application")
    # Try and except should the camera not have been used within the system.
    try:
        # Stop the frame capture
        cap.release()
        # Destroy the all the CV2 windows
        cv2.destroyAllWindows()
    except:
        print("Image capture not used")