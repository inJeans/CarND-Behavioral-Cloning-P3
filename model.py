import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
import pandas as pd
# tf.python.control_flow_ops = tf
from keras import models, optimizers, backend
from keras.layers import core, convolutional, pooling, Lambda, Cropping2D
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from sklearn import model_selection
from data import generate_samples, preprocess
# from weights_logger_callback import WeightsLogger

local_project_path = './'
local_data_path = os.path.join(local_project_path, 'data')

BATCH_SIZE = 1024


if __name__ == '__main__':
    # Read the data
    print("Loading data")
    car_images = np.load("car_images.npy")
    steering_angles = np.load("steering_angles.npy")
    # Split data into training and validation sets
    X_train, X_test, y_train, y_test = model_selection.train_test_split(car_images,
                                                                        steering_angles,
                                                                        test_size=.2)
    print("There are {} total samples".format(len(car_images)))

    # datagen = ImageDataGenerator(featurewise_center=True,
    #                              featurewise_std_normalization=True,
    #                              fill_mode="constant")
    # print("Fitting generator")
    # datagen.fit(X_train)

    # Model architecture
    print("Compiling model")
    if os.path.isfile("model-correct.h5"):
        model = load_model('model-correct.h5')
    else:
        model = models.Sequential()
        model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
        model.add(Cropping2D(cropping=((60, 25), (0, 0)), input_shape=(160, 320, 3)))
        model.add(convolutional.Convolution2D(16, (3, 3), activation='relu'))
        model.add(pooling.MaxPooling2D(pool_size=(2, 2)))
        model.add(convolutional.Convolution2D(32, (3, 3), activation='relu'))
        model.add(pooling.MaxPooling2D(pool_size=(2, 2)))
        model.add(convolutional.Convolution2D(64, (3, 3), activation='relu'))
        model.add(pooling.MaxPooling2D(pool_size=(2, 2)))
        model.add(core.Flatten())
        model.add(core.Dense(500, activation='relu'))
        model.add(core.Dropout(.5))
        model.add(core.Dense(100, activation='relu'))
        model.add(core.Dropout(.25))
        model.add(core.Dense(20, activation='relu'))
        model.add(core.Dense(1))
        model.compile(optimizer=optimizers.Adam(lr=1e-04),
                      loss='mean_squared_error',
                      metrics=['mae',])

    # for _ in range(10):
    #     history = model.fit_generator(
    #         generate_samples(df_train, local_data_path),
    #         samples_per_epoch=df_train.shape[0],
    #         nb_epoch=1,
    #         validation_data=generate_samples(df_valid, local_data_path, augment=False),
    #         # callbacks=[WeightsLogger(root_path=local_project_path)],
    #         nb_val_samples=df_valid.shape[0],
    #     )

    #     model.save('model.h5')

    print("Training")
    for _ in range(10):
        history = model.fit(X_train, 
                            y_train, 
                            batch_size=BATCH_SIZE,
                            epochs=20,
                            verbose=True,
                            validation_data=(X_test, y_test)
                           )

        model.save('model-correct.h5')

    with open(os.path.join(local_project_path, 'model.json'), 'w') as file:
        file.write(model.to_json())

    backend.clear_session()