import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.interpolate import splprep, splev
from PIL import Image

def translate_spline(out, out_dx_dy, translation_factor = 1):
  # Translating each point along the spline
  translated_points = []
  for dx, dy, x, y in zip(*out_dx_dy, *out):
    mag = np.sqrt(dx**2 + dy**2)
    ux, uy = -dy / mag, dx / mag
    new_x = x + ux * translation_factor
    new_y = y + uy * translation_factor
    translated_points.append((new_x, new_y))

  translated_points = np.array(translated_points)
  return translated_points

def draw_points(image, shape, x_coords, y_coords, color):
    for x, y in zip(x_coords, y_coords):
        if 0 <= x < shape[1] and 0 <= y < shape[0]:
            image[int(y), int(x)] = color

def get_pixel_data(file_path):
    segment = mpimg.imread(file_path)
    img_shape = np.shape(segment)
    threshold_value = 0.25
    binary_segment = (segment > threshold_value).astype(np.float32)
    binary_segment_array = np.array(binary_segment)
    #Locate Pixel Centres
    pixels = np.argwhere(binary_segment_array == np.amax(binary_segment_array))
    x_coords = []
    y_coords = []
    for coord in pixels:
       x_coords.append(coord[1])
       y_coords.append(coord[0])
    xy_coords = list(zip(x_coords, y_coords))
    xy_coords.sort()
    xy_data = np.array(xy_coords)
    x_data, y_data = xy_data.T
    return x_data, y_data, img_shape

def param_spline(x_data, y_data, smoothing_factor, order):
    tck, u = splprep([x_data, y_data], s=smoothing_factor, k=order)
    unew = np.linspace(0, 1, 900)
    out = splev(unew, tck)
    out_dx_dy = splev(unew, tck, der=1)
    return out, out_dx_dy

def split_segment(x_data, y_data, num_segments=3):
    # Ensure x_data and y_data are numpy arrays for easier slicing
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    
    # Calculate the size of each segment
    total_points = len(x_data)
    points_per_segment = total_points // num_segments  # Use floor division to ensure an integer result
    
    # Initialize lists to hold the segment data
    segments_x = []
    segments_y = []
    
    # Create each segment
    for i in range(num_segments):
        # Calculate the start and end indices for the current segment
        start_idx = i * points_per_segment
        # For the last segment, go all the way to the end of the data array
        end_idx = (i + 1) * points_per_segment if i < num_segments - 1 else total_points
        
        # Extract the segment and add it to the lists
        segment_x = x_data[start_idx:end_idx]
        segment_y = y_data[start_idx:end_idx]
        
        segments_x.append(segment_x)
        segments_y.append(segment_y)
    
    return segments_x, segments_y
   
   
   














