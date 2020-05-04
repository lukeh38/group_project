# IMPORTS
import os
import numpy as np
import tensorflow.compat.v1 as tf
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.models import load_model
import ResNet
import tkinter as tk
from tkinter import filedialog


# Function for the forward propagation used for the prediction
def forward_propagation(x, params):

    # Retrieve the parameters from the dictionary "parameters"
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']
    W3 = params['W3']
    b3 = params['b3']

    # Implements the forward propagation method of: Linear -> Relu -> Linear -> Relu -> Linear -> Softmax
    Z1 = tf.add(tf.matmul(W1, x), b1)
    A1 = tf.nn.relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)
    A2 = tf.nn.relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)

    return Z3
# END FORWARD_PROPAGATION


# Function for processing the input image
def predict(target, params):

    W1 = tf.convert_to_tensor(params["W1"])
    b1 = tf.convert_to_tensor(params["b1"])
    W2 = tf.convert_to_tensor(params["W2"])
    b2 = tf.convert_to_tensor(params["b2"])
    W3 = tf.convert_to_tensor(params["W3"])
    b3 = tf.convert_to_tensor(params["b3"])

    params = {"W1": W1,
              "b1": b1,
              "W2": W2,
              "b2": b2,
              "W3": W3,
              "b3": b3}

    x = tf.placeholder("float", [12288, 1])

    z3 = forward_propagation(x, params)
    p = tf.argmax(z3)

    sess = tf.Session()
    prediction = sess.run(p, feed_dict={x: target})

    return prediction
# END PREDICT


# Model tries to predict the Hand Sign shown in the target image
def test_model(target_img):

    img_path = target_img
    img = image.load_img(img_path, target_size=(64, 64))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    print('Input image shape:', x.shape)
    print("Model's Predicted Hand Sign =\n[(0), (1), (2), (3), (4), (5)]")
    print(np.around(model.predict(x)))  # Outputs the models predicted hand sign in a binary format
# END TEST_MODEL()


# Checks if the required Training Model exists
if os.path.isfile("HandSignResNetModel.h5"):
    model = load_model("HandSignResNetModel.h5")

else:
    ResNet.train_model()
    model = load_model("HandSignResNetModel.h5")


# An open file dialog will pop up, asking for the image file
# It then takes file path of the image and passes it to test_model()
root = tk.Tk()
root.withdraw()
img_file = filedialog.askopenfilename(initialdir="sample_images", title="Select file", filetypes=(("JPEG image file", "*.jpg"), ("All files", "*.*")))
test_model(img_file)
