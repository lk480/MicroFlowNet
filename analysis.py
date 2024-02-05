import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import matlab.engine
import cv2
from PIL import Image
from segmentation_model import load_hvi_image, SA_UNet
from translation import get_pixel_data, param_spline, translate_spline, draw_points

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
binary_image = eng.central_kymograph_generation(segmentation_file_path, image_sequence_dir)
eng.quit()

x_data, y_data, img_shape = get_pixel_data(segment_file_path)
height, width = img_shape

#Generate Profile Kymographs 
eng = matlab.engine.start_matlab()
for t_factor in np.linspace(-4, 4, 3):
    #Fit Parametric Spline and Translate
    out, out_dx_dy = param_spline(x_data, y_data, smoothing_factor = 8, order = 2)
    translated_points = translate_spline(out, out_dx_dy, translation_factor = t_factor)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    pos_spline_coords = np.unique(np.ceil(translated_points).astype(int), axis=0)
    draw_points(image, img_shape, pos_spline_coords[:, 0], pos_spline_coords[:, 1], color=[255, 255, 255])
    img = Image.fromarray(image)
    img.save(translated_segment_file_path)
    print(np.round(t_factor, 1))
    binary_image = eng.variable_axis_kymograph_generation(translated_segment_file_path, image_sequence_dir, t_factor)
eng.quit()