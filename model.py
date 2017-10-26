import numpy as np
import os
import tensorflow as tf
from keras import models, optimizers, backend
from keras.layers import core, convolutional, pooling, Lambda, Cropping2D
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from sklearn import model_selection
from data import generate_samples, preprocess

PROJECT_PATH = './'
DATA_PATH = os.path.join(local_project_path, 'data')

DEAFULT_MODEL = "model.h5"

BATCH_SIZE = 1024


def main():
    model = generate_model()

    image_filepath = os.pth.join(DATA_PATH,
                                 "car_images.npy")
    steering_filepath = os.pth.join(DATA_PATH,
                                 "steering_angles.npy")
    if os.path.isfile(image_filepath) and os.path.isfile(steering_filepath):
        # Read the data
        print("Loading data")
        car_images = np.load(image_filepath)
        steering_angles = np.load(steering_filepath )
        # Split data into training and validation sets
        X_train, X_test, y_train, y_test = model_selection.train_test_split(car_images,
                                                                            steering_angles,
                                                                            test_size=.2)
        print("There are {} total samples".format(len(car_images)))

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
    else:
        # Read the data
        driving_log_path = os.path.join(DATA_PATH,
                                        "driving_log.csv")
        df = pd.io.parsers.read_csv(driving_log_path)
        # Split data into training and validation sets
        df_train, df_valid = model_selection.train_test_split(df, test_size=.2)

        datagen = ImageDataGenerator(featurewise_center=True,
                                     featurewise_std_normalization=True,
                                     fill_mode="constant")
        # print("Fitting generator")
        # datagen.fit(X_train)

        for _ in range(10):
            history = model.fit_generator(
                generate_samples(df_train, DATA_PATH),
                samples_per_epoch=df_train.shape[0],
                nb_epoch=1,
                validation_data=generate_samples(df_valid, DATA_PATH, augment=False),
                nb_val_samples=df_valid.shape[0],
            )

            model.save('model.h5')


    with open(os.path.join(local_project_path, 'model.json'), 'w') as file:
        file.write(model.to_json())

    backend.clear_session()

def generate_model(model_filename=DEAFULT_MODEL):
    model = None

    # Model architecture
    if os.path.isfile(model_filename):
        print("Loading model")
        model = load_model(model_filename)
    else:
        print("Compiling model")
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

    return model

if __name__ == '__main__':
    main()
