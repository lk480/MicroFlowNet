import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import linregress

# Load the image
log_magnitude_image = cv2.imread('/Users/lohithkonathala/iib_project/synthetic_kymographs/Angle_120.5/log_magnitude_120.5_instance_1.png', 0)
plt.imshow(log_magnitude_image, cmap='gray')

_, thresh_image = cv2.threshold(log_magnitude_image, 70, 255, cv2.THRESH_BINARY)
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

# Plot the filtered data
plt.scatter(x_filtered, y_filtered, color='blue')

# Plot the regression line
plt.plot(x_filtered, regression_line, color='red', label=f'y = {slope:.2f}x + {intercept:.2f}')

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






