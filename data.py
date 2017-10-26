import numpy as np
import skimage.transform as sktransform
import random
import matplotlib.image as mpimg
import os


# Cameras we will use
cameras = ['left', 'center', 'right']
cameras_steering_correction = [.25, 0., -.25]

# def preprocess(image, top_offset=.375, bottom_offset=.125):
#     """
#     Applies preprocessing pipeline to an image: crops `top_offset` and `bottom_offset`
#     portions of image, resizes to 32x128 px and scales pixel values to [0, 1].
#     """
#     top = int(top_offset * image.shape[0])
#     bottom = int(bottom_offset * image.shape[0])
#     image = sktransform.resize(image[top:-bottom, :], (32, 128, 3), mode="constant")
#     return image

def generate_samples(data, root_path, augment=True):
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
            x = np.empty([0, 32, 128, 3], dtype=np.float32)
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
            # Randomly flip half of images in the batch
            flip_indices = random.sample(range(x.shape[0]), int(x.shape[0] / 2))
            x[flip_indices] = x[flip_indices, :, ::-1, :]
            y[flip_indices] = -y[flip_indices]
            yield (x, y)