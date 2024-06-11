import tensorflow as tf
from tensorflow import keras

def create_cnn_model(input_shape, num_classes):
    model = keras.Sequential()

    model.add(keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPool2D(2,2))

    model.add(keras.layers.Conv2D(64, (3,3), activation='relu'))
    model.add(keras.layers.MaxPool2D(2,2))

    model.add(keras.layers.Conv2D(128, (3,3), activation='relu'))
    model.add(keras.layers.MaxPool2D(2,2))

    model.add(keras.layers.Conv2D(128, (3,3), activation='relu'))
    model.add(keras.layers.MaxPool2D(2,2))

    model.add(keras.layers.Flatten())

    model.add(keras.layers.Dense(512, activation='relu'))

    model.add(keras.layers.Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model
