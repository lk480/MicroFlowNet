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
    y_data, x_data = pixels.T
    return x_data, y_data, img_shape

def param_spline(x_data, y_data, smoothing_factor = 8, order = 1):
    tck, u = splprep([x_data, y_data], s=smoothing_factor, k=order)
    unew = np.linspace(0, 1, 1500)
    out = splev(unew, tck)
    out_dx_dy = splev(unew, tck, der=1)
    return out, out_dx_dy














