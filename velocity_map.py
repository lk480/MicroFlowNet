import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import matlab.engine
import cv2
from PIL import Image
from segmentation_model import load_hvi_image, SA_UNet
from auto_texture import generate_FFT, window_function, histogram_of_gradients
import matplotlib.image as mpimg
from velocity_estimation import est_central_velocity, get_vessel_index


#Specify Directories
hvi_file_path = '/Users/lohithkonathala/Downloads/MII_start_0_end_9.pgm'
segmentation_file_path = '/Users/lohithkonathala/iib_project/vessel_segmentation.png'
image_sequence_dir = '/Users/lohithkonathala/Documents/IIB Project/affine_registered_sequences/willeye_affine/'
segment_file_path = '/Users/lohithkonathala/iib_project/vessel_segment.png'
translated_segment_file_path = '/Users/lohithkonathala/iib_project/translated_vessel_segment.png'
weight_file_path = '/Users/lohithkonathala/iib_project/sa_unet_CHASE_weights.h5'

#Perform Vessel Segmentation 
hvi_image = load_hvi_image(hvi_file_path)
model = SA_UNet()
model.load_weights(weight_file_path)
hvi_image_batched = tf.expand_dims(hvi_image, axis=0)

if np.shape(hvi_image_batched)[0] is None:
    print("ERROR: Data has not been batched")
elif np.shape(hvi_image_batched)[3] != 3:
    print("ERROR: Image must be 3-channel")
else:
    prediction = model.predict(hvi_image_batched)
    predicted_segmentation = prediction[0]*255.0
    print(np.shape(predicted_segmentation))
    _, predicted_segmentation_thresholded = cv2.threshold(predicted_segmentation, 100, 255, cv2.THRESH_BINARY)
    cv2.imwrite(segmentation_file_path, predicted_segmentation)


#Generate Central Axis Kymograph 
eng = matlab.engine.start_matlab()
for vessel_index in range(1,5):
    binary_image = eng.central_kymograph_generation(segmentation_file_path, image_sequence_dir, vessel_index)
eng.quit()

files = os.listdir('/Users/lohithkonathala/iib_project/central_axis_kymographs')
sorted_files = sorted(files, key=get_vessel_index)
axial_flow_velocities = []

for file_name in sorted_files:
    print(f"Vessel Index {get_vessel_index(file_name)}")
    file_path = os.path.join('/Users/lohithkonathala/iib_project/central_axis_kymographs', file_name)
    flow_velocity = est_central_velocity(file_path, scale_factor = 3.676, time_factor = 1.667)
    axial_flow_velocities.append(flow_velocity)

print(axial_flow_velocities)





