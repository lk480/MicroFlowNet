import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import linregress
from auto_texture import get_translation_factor, generate_FFT, window_function
import matplotlib.image as mpimg
import os

# Load the image
files = os.listdir('/Users/lohithkonathala/iib_project/kymographs_will')
sorted_files = sorted(files, key=get_translation_factor)
file_name = sorted_files[0]

print(f"Translation Factor {get_translation_factor(file_name)}")
file_path = os.path.join('/Users/lohithkonathala/iib_project/kymographs_will', file_name)
image = mpimg.imread(file_path, cv2.IMREAD_GRAYSCALE)
print(np.shape(image))

#Spatial FFT
windowed_image = window_function(image)
fft_shifted, magnitude_spectrum, log_magnitude_spectrum = generate_FFT(image)
height, width = np.shape(log_magnitude_spectrum)
log_magnitude_spectrum = cv2.resize(log_magnitude_spectrum, (width, height), interpolation=cv2.INTER_CUBIC)


max_val = np.max(log_magnitude_spectrum)
min_val = np.min(log_magnitude_spectrum)
scaled_spectrum = ((log_magnitude_spectrum - min_val) / (max_val - min_val)) * 255
scaled_spectrum = scaled_spectrum.astype(np.uint8)

plt.imshow(scaled_spectrum)
plt.show()

_, thresh_image = cv2.threshold(scaled_spectrum, 110, 255, cv2.THRESH_BINARY)
plt.imshow(thresh_image, cmap='gray')
plt.show()

# Find the coordinates of white pixels
y, x = np.where(thresh_image == 255)

# Combine x and y into a single array and sort by x
points = np.column_stack((x, y))
points_sorted = points[np.argsort(points[:, 0])]

# Dictionary to hold the x-values and the corresponding unique y-values (up to 3)
# Your existing code
unique_x_to_y = {}

for x_val, y_val in points_sorted:
    if x_val not in unique_x_to_y:
        unique_x_to_y[x_val] = [y_val]
    else:
        if y_val not in unique_x_to_y[x_val] and len(unique_x_to_y[x_val]) < 3:
            unique_x_to_y[x_val].append(y_val)

threshold = 5  # Set the threshold for what you would consider a horizontal line

# Step 2: Count y-values across all x-values
y_counts = {}
for y_vals in unique_x_to_y.values():
    for y_val in y_vals:
        y_counts[y_val] = y_counts.get(y_val, 0) + 1

# Step 3: Filter out y-values that exceed the threshold
filtered_x_to_y = {}
for x_val, y_vals in unique_x_to_y.items():
    filtered_x_to_y[x_val] = [y_val for y_val in y_vals if y_counts[y_val] <= threshold]

# Flatten the list of y-values for plotting, keeping the association with x-values
x_filtered = []
y_filtered = []
for x_val, y_vals in filtered_x_to_y.items():
    for y_val in y_vals:
        x_filtered.append(x_val)
        y_filtered.append(y_val)

x_filtered = np.array(x_filtered)
y_filtered = np.array(y_filtered)

# Plot the filtered data
plt.scatter(x_filtered, y_filtered, color='blue')
plt.xlim(0, 128)
plt.ylim(0, 128)

plt.xlabel('X-Value')
plt.ylabel('Y-Value')
plt.gca().set_aspect('equal', adjustable='box')
plt.show()


slope, intercept, r_value, p_value, std_err = linregress(x_filtered, y_filtered)

# Use the slope and intercept to calculate the points of the regression line
regression_line = slope * x_filtered + intercept

#Use intial estimate of regression line to compute residuals
res = list(y_filtered - regression_line)
threshold = 0.75*np.std(res)

data = list(zip(x_filtered, y_filtered))

outlier_indices = []
for residual in res:
    if np.abs(residual) > threshold:
        outlier_indices.append(res.index(residual))

outlier_indices = set(outlier_indices)
thresholded_filtered_data = [t for i, t in enumerate(data) if i not in outlier_indices]
x_thresholded_filtered, y_thresholded_filtered = zip(*thresholded_filtered_data)
x_thresholded_filtered = np.array(x_thresholded_filtered)
y_thresholded_filtered = np.array(y_thresholded_filtered)


slope, intercept, r_value, p_value, std_err = linregress(x_thresholded_filtered, y_thresholded_filtered)

# Use the slope and intercept to calculate the points of the regression line
regression_line = slope * x_thresholded_filtered + intercept

# Plot the filtered data
plt.scatter(x_thresholded_filtered, y_thresholded_filtered, color='blue')

# Plot the regression line
plt.plot(x_thresholded_filtered, regression_line, color='red', label=f'y = {slope:.2f}x + {intercept:.2f}')

# Add labels and legend
plt.xlim(0, 128)
plt.ylim(0, 128)
plt.xlabel('X-Value')
plt.ylabel('Y-Value')
plt.gca().set_aspect('equal', adjustable='box')
plt.legend()

# Show the plot
plt.show()

# Print the slope of the line
print(f"Main orientation of texture is: {np.arctan(np.abs(slope)) * (180/np.pi) + 90}")






