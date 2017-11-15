import argparse
import numpy as np
import skimage.transform as sktransform
import random
import matplotlib.image as mpimg
import os

# Project globals
DATA_DIR = "/flush1/wat421/Behavioral_Cloning/data"
CSV_PATH = "/flush1/wat421/Behavioral_Cloning/data/driving_log.csv"

# Cameras we will use
CAMERAS = ['left', 'center', 'right']
STEERING_CORRECTION = [.25, 0., -.25]

def preprocess(csv_path=CSV_PATH,
               data_dir=DATA_DIR):
    data_df = pd.read_csv(csv_path)
    # data_df["abs_steering"] = data_df["steering"].abs()

    # data_df.hist(column="abs_steering",
    #              bins=100)
    # plt.axis([0, 1, 0, 500])
    # plt.savefig("test.png")

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

    image_filepath = os.path.join(data_dir,
                                  "car_images.npy")
    np.save(image_filepath,
            car_images)
    steering_filepath = os.path.join(data_dir,
                                     "steering_angles.npy")
    np.save(steering_filepath,
            steering_angles)

    return

def generate_samples(data, root_path, batch_size=128, augment=True):
    """
    Keras generator yielding batches of training/validation data.
    Applies data augmentation pipeline if `augment` is True.
    """
    while True:
        # Generate random batch of indices
        indices = np.random.permutation(data.count()[0])
        batch_size = 128
        for batch in range(0, len(indices), batch_size):
            batch_indices = indices[batch:(batch + batch_size)]
            # Output arrays
            x = np.empty([0, 160, 320, 3], dtype=np.float32)
            y = np.empty([0], dtype=np.float32)
            # Read in and preprocess a batch of images
            for i in batch_indices:
                # Randomly select camera
                camera = np.random.randint(len(cameras)) if augment else 1
                # Read frame image and work out steering angle
                image_path = os.path.join(root_path,
                                          data[cameras[camera]].values[i].strip())
                image = mpimg.imread(image_path)
                angle = data.steering.values[i] + cameras_steering_correction[camera]
                # Append to batch
                x = np.append(x, [image], axis=0)
                y = np.append(y, [angle])
            # Randomly flip half of images in the batch
            flip_indices = random.sample(range(x.shape[0]), int(x.shape[0] / 2))
            x[flip_indices] = x[flip_indices, :, ::-1, :]
            y[flip_indices] = -y[flip_indices]
            yield (x, y)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Augment input data.')
    parser.add_argument('--data',
                        type=str,
                        default=DATA_DIR,
                        help='path to your training data')
    parser.add_argument('--csv',
                        type=str,
                        default=CSV_PATH,
                        help='path to your csv file')

    args = parser.parse_args()

    preprocess(csv_path=args.csv,
               data_dir=args.data)