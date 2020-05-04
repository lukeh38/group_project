# IMPORTS
import h5py
import numpy as np
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, \
    AveragePooling2D, MaxPooling2D
from keras.models import Model
from keras.initializers import glorot_uniform

import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)


# Loading and preparing the Dataset
def data_prep():

    # Retrieve the Training Data - features (x) and labels (y)
    train_dataset = h5py.File('handsigns_dataset/train_signs.h5', "r")
    x_train = np.array(train_dataset["train_set_x"][:])
    y_train = np.array(train_dataset["train_set_y"][:])

    # Retrieve the Testing Data - features (x) and labels (y)
    test_dataset = h5py.File('handsigns_dataset/test_signs.h5', "r")
    x_test = np.array(test_dataset["test_set_x"][:])
    y_test = np.array(test_dataset["test_set_y"][:])

    # Retrieve the List of Classes
    classes = np.array(test_dataset["list_classes"][:])

    # Reshape the labels, making it easier to feed into the model
    y_train = y_train.reshape((1, y_train.shape[0]))
    y_test = y_test.reshape((1, y_test.shape[0]))

    return x_train, y_train, x_test, y_test, classes
# END DATA_PREP()


# Identity Block, a standard function for Residual Neural Networks (ResNet)
# Used for when the input and output activations have the same dimensions
def identity_block(x, f, filters, stage, block):

    # Define variable names to be used for the components of the main path
    conv_base_name = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Define filters
    F1, F2, F3 = filters

    # Retain the input value, allowing it to be added back to the main path
    x_shortcut = x

    # First Component of the main path
    x = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid',
               name=conv_base_name + '2a', kernel_initializer=glorot_uniform(seed=0))(x)
    x = BatchNormalization(axis=3, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    # Second Component
    x = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same',
               name=conv_base_name + '2b', kernel_initializer=glorot_uniform(seed=0))(x)
    x = BatchNormalization(axis=3, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    # Third Component
    x = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid',
               name=conv_base_name + '2c', kernel_initializer=glorot_uniform(seed=0))(x)
    x = BatchNormalization(axis=3, name=bn_name_base + '2c')(x)

    # Add a shortcut value to the main path which will be passed through the RELU activation
    x = Add()([x, x_shortcut])
    x = Activation('relu')(x)

    return x
# END IDENTITY_BLOCK()


# Convolutional Block, another standard function in ResNet
# Used for when the dimensions of the input and output activations do not match up
def convolutional_block(x, f, filters, stage, block, s=2):

    # Define variable names to be used for the components of the main path
    conv_base_name = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Define filters
    F1, F2, F3 = filters

    # Retain the input value, allowing it to be added back to the main path
    x_shortcut = x

    # First Component of the main path
    x = Conv2D(F1, (1, 1), strides=(s, s), name=conv_base_name + '2a',
               kernel_initializer=glorot_uniform(seed=0))(x)
    x = BatchNormalization(axis=3, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    # Second Component
    x = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same',
               name=conv_base_name + '2b', kernel_initializer=glorot_uniform(seed=0))(x)
    x = BatchNormalization(axis=3, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    # Third Component
    x = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid',
               name=conv_base_name + '2c', kernel_initializer=glorot_uniform(seed=0))(x)
    x = BatchNormalization(axis=3, name=bn_name_base + '2c')(x)

    # Shortcut Path
    x_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid',
                        name=conv_base_name + '1', kernel_initializer=glorot_uniform(seed=0))(x_shortcut)
    x_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(x_shortcut)

    # Add a shortcut value to the main path and pass it through the RELU activation
    x = Add()([x, x_shortcut])
    x = Activation('relu')(x)

    return x
# END CONVOLUTIONAL_BLOCK()


# Builds the ResNet50
def ResNet50(input_shape=(64, 64, 3), classes=6):

    # Define the input shape as a tensor with the shape 'input_shape'
    x_input = Input(input_shape)

    # Define the Zero-Padding
    x = ZeroPadding2D((3, 3))(x_input)

    # First Stage
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1',
               kernel_initializer=glorot_uniform(seed=0))(x)
    x = BatchNormalization(axis=3, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # Second Stage
    x = convolutional_block(x, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    # Third Stage
    x = convolutional_block(x, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    # Fourth Stage
    x = convolutional_block(x, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    # Fifth Stage
    x = convolutional_block(x, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    # Average Pooling
    x = AveragePooling2D((2, 2), name="avg_pool")(x)

    # Output Layer
    x = Flatten()(x)
    x = Dense(classes, activation='softmax', name='fc' + str(classes),
              kernel_initializer=glorot_uniform(seed=0))(x)

    # Creating the Model with placeholders
    model = Model(inputs=x_input, outputs=x, name='ResNet50')

    return model
# END RESNET50


# Function for converting the labels into one-hot matrices
def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y
# END CONVERT_TO_ONE_HOT()


# Train the ResNet50 model
def train_model():

    # Configure the learning process and compile the model
    model = ResNet50(input_shape=(64, 64, 3), classes=6)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Retrieve the prepared training and testing data
    train_x, train_y, test_x, test_y, class_list = data_prep()

    # Normalise the training images
    train_samples = train_x / 255
    test_samples = test_x / 255

    # Convert the labels from both the training and testing set into one-hot matrices
    train_labels = convert_to_one_hot(train_y, 6).T
    test_labels = convert_to_one_hot(test_y, 6).T

    # Train and Evaluate the model
    history = model.fit(train_samples, train_labels, epochs=25, batch_size=32)  # Change the number of epochs here

    test_loss, test_acc = model.evaluate(test_samples, test_labels)
    print("Loss = " + str(test_loss))
    print("Test Accuracy = " + str(test_acc))

    model.save("HandSignResNetModel.h5")
# END TRAIN_MODEL()


"""
# For testing purposes...
print("Number of Training Samples: " + str(train_samples.shape[0]))
print("Number of Testing Samples: " + str(test_samples.shape[0]))
print ("x_train shape: " + str(train_samples.shape))
print ("Y_train shape: " + str(train_labels.shape))
print ("x_test shape: " + str(test_samples.shape))
print ("Y_test shape: " + str(test_labels.shape))
"""



