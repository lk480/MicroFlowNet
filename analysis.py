import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import matlab.engine
import cv2
from PIL import Image
from segmentation_model import load_hvi_image, SA_UNet
from translation import get_pixel_data, param_spline, translate_spline, draw_points, split_segment
from velocity_estimation import generate_velocity_profile

"""Specify Directories"""
hvi_file_path = '/Users/lohithkonathala/Documents/IIB Project/12x_rigid_body_final_denoised/MII_start_0_end_9.pgm'
segmentation_file_path = '/Users/lohithkonathala/iib_project/vessel_segmentation.png'
image_sequence_dir = '/Users/lohithkonathala/Documents/IIB Project/12x_rigid_body_final_denoised'
segment_file_path = '/Users/lohithkonathala/iib_project/vessel_segment.png'
sub_segment_file_path = '/Users/lohithkonathala/iib_project/vessel_sub_segment.png'
translated_segment_file_path = '/Users/lohithkonathala/iib_project/translated_vessel_segment.png'
weight_file_path = '/Users/lohithkonathala/iib_project/sa_unet_CHASE_weights.h5'

analysis = True

if analysis == True:
    """Vessel Segmentation"""
    hvi_image = load_hvi_image(hvi_file_path) #Load HVI Image using load_hvi_image helper function
    model = SA_UNet() #Choose segmentation model i.e. SA_UNet
    model.load_weights(weight_file_path) #Load model weights 
    hvi_image_batched = tf.expand_dims(hvi_image, axis=0) #Introduce batch dimensions for compatibility with CNN input layer

    if np.shape(hvi_image_batched)[0] is None:
        print("ERROR: Data has not been batched") 
        raise ValueError("Image must have batch dimensions")
    elif np.shape(hvi_image_batched)[3] != 3:
        raise ValueError("Image Must be 3 Channel")
    else:
        prediction = model.predict(hvi_image_batched) #Segmentation 
        predicted_segmentation = prediction[0]*255.0
        #Apply a threshold to remove noise from the segmentation
        _, predicted_segmentation_thresholded = cv2.threshold(predicted_segmentation, 110, 255, cv2.THRESH_BINARY)  
        cv2.imwrite(segmentation_file_path, predicted_segmentation)

    """Generate Central Axis Kymograph i.e. Determine Axial Flow Velocity"""
    eng = matlab.engine.start_matlab()
    binary_image = eng.central_kymograph_generation(segmentation_file_path, image_sequence_dir, 1) 
    eng.quit()

    """Extract Pixel Data from Vessel of Interest"""
    x_data, y_data, img_shape = get_pixel_data(segment_file_path)
    height, width = img_shape

    """Generate Translated Axis Kymographs i.e. Determine Velocity Variation Across Vessel Diameter  """
    eng = matlab.engine.start_matlab()
    for t_factor in np.linspace(-7, 7, 20): #Set bounds for the translation factor 
        out, out_dx_dy = param_spline(x_data, y_data, smoothing_factor = 8, order = 2) #Fit Parametric Spline
        translated_points = translate_spline(out, out_dx_dy, translation_factor = t_factor) #Translate Spline 
        image = np.zeros((height, width, 3), dtype=np.uint8) #Create empty image with matching dimensions 
        pos_spline_coords = np.unique(np.ceil(translated_points).astype(int), axis=0)  #Converts Translated Points to Integer Values 
        draw_points(image, img_shape, pos_spline_coords[:, 0], pos_spline_coords[:, 1], color=[255, 255, 255]) #Label Pixels on an Blank Image
        img = Image.fromarray(image)
        img.save(translated_segment_file_path) #Save the translated segment for visual verificiation 
        print(np.round(t_factor, 1)) 
        binary_image = eng.variable_axis_kymograph_generation(translated_segment_file_path, image_sequence_dir, t_factor) #Generate Kymograph for Translated Segment
    eng.quit()