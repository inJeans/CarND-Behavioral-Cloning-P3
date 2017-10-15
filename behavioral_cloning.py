from __future__ import print_function

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import os

from skimage.io import imread

import numpy as np
import pandas as pd
import seaborn as sns

DATA_DIR = "/flush1/wat421/Behavioral_Cloning/data"
CSV_PATH = "/flush1/wat421/Behavioral_Cloning/data/driving_log.csv"

COLUMN_NAMES = ["image", "throttle"]

CAMERA_OFFSETS = {
                  # "left" : 0.2, 
                  "center" : 0., 
                  # "right" : -0.2,
                  }

def main():

    data_df = pd.read_csv(CSV_PATH)
    data_df["abs_steering"] = data_df["steering"].abs()

    data_df.hist(column="abs_steering",
                 bins=100)
    plt.axis([0, 1, 0, 500])
    plt.savefig("test.png")

    car_images = []
    steering_angles = []
    for i, image in data_df.iterrows():
        for camera, offset in CAMERA_OFFSETS.items():
            image_path = os.path.join(DATA_DIR, image[camera].strip())
            image_pixels = np.asarray(imread(image_path))
            steering = image.steering + offset

            car_images.append(image_pixels)
            steering_angles.append(steering)

            # flip image
            car_images.append(np.fliplr(image_pixels))
            steering_angles.append(-1.*steering)

        if i % 100 == 0:
            print("{} \% complete".format(i/len(data_df)*100.))

    np.save("car_images",
            car_images)
    np.save("steering_angles",
            steering_angles)

    return

if __name__ == '__main__':
    main()
