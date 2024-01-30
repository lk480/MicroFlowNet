import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import matlab.engine
import cv2
from segmentation import load_hvi_image, SA_UNet

file_path = '/Users/lohithkonathala/Downloads/MII_start_0_end_9.pgm'
segmentation_file_path = '/Users/lohithkonathala/iib_project/vessel_segmentation.png'
image_sequence_dir = '/Users/lohithkonathala/Documents/IIB Project/affine_registered_sequences/willeye_affine/'

hvi_image = load_hvi_image(file_path)
model = SA_UNet()
model.load_weights('/Users/lohithkonathala/iib_project/sa_unet_CHASE_weights.h5')
hvi_image_batched = tf.expand_dims(hvi_image, axis=0)
#Check Dimensions
if np.shape(hvi_image_batched)[0] is None:
    print("ERROR: Data has not been batched")
elif np.shape(hvi_image_batched)[3] != 3:
    print("ERROR: Image must be 3-channel")
else:
    prediction = model.predict(hvi_image_batched)
    predicted_segmentation = prediction[0]*255.0
    print(np.shape(predicted_segmentation))
    plt.imshow(predicted_segmentation, cmap='gray')
    cv2.imwrite(segmentation_file_path, predicted_segmentation)


eng = matlab.engine.start_matlab()
binary_image = eng.kymograph_generation(segmentation_file_path, image_sequence_dir)
eng.quit()