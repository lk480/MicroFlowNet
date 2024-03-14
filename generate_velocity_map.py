import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import matlab.engine
import cv2
from PIL import Image
from segmentation_model import load_hvi_image, SA_UNet
import matplotlib.image as mpimg
from velocity_estimation import estimate_axial_velocity


#Specify Directories
hvi_file_path = '/Users/lohithkonathala/Documents/IIB Project/12x_rigid_body/MII_start_0_end_9.pgm'
segmentation_file_path = '/Users/lohithkonathala/iib_project/vessel_segmentation.png'
image_sequence_dir = '/Users/lohithkonathala/Documents/IIB Project/12x_rigid_body'
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
vessels_of_interest = [1, 2, 4, 5, 6, 9]
for vessel_index in vessels_of_interest:
    binary_image = eng.central_kymograph_generation(segmentation_file_path, image_sequence_dir, vessel_index)
eng.quit()

central_kymo_dir = '/Users/lohithkonathala/iib_project/central_axis_kymographs'

upper_bound_velocities, median_velocities, lower_bound_velocities, vessel_indices =  estimate_axial_velocity(central_kymo_dir)

eng = matlab.engine.start_matlab()
matlab_velocities = matlab.double(median_velocities)
matlab_vessels_of_interest = matlab.double(vessels_of_interest)
velocity_map = eng.visualise_velocity(segmentation_file_path, image_sequence_dir, matlab_velocities, matlab_vessels_of_interest)
eng.quit()



