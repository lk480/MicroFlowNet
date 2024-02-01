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

#Load Vessel Segment
segment =  mpimg.imread('/Users/lohithkonathala/iib_project/vessel_segment.png')
height, width = np.shape(segment)
#Convert Segment Image to Binary
threshold_value = 0.25
binary_segment = (segment > threshold_value).astype(np.float32)
binary_segment_array = np.array(binary_segment)
#Locate Pixel Centres
pixels = np.argwhere(binary_segment_array == np.amax(binary_segment_array))
y_data, x_data = pixels.T
#Fit Parametric Spline
tck, u = splprep([x_data, y_data], s=8, k=1)
unew = np.linspace(0, 1, 1500)
out = splev(unew, tck)
out_dx_dy = splev(unew, tck, der=1)
#Translation 
translated_points = translate_spline(out, out_dx_dy, translation_factor = 12)
pos_spline_coords = np.unique(np.ceil(translated_points).astype(int), axis=0)
#Create New Vessel Segment
image = np.zeros((height, width, 3), dtype=np.uint8)
draw_points(image, (height, width), pos_spline_coords[:, 0], pos_spline_coords[:, 1], color=[0, 255, 0])  
img = Image.fromarray(image)
plt.imshow(img)
plt.axis('off')
plt.show()



















