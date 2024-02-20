import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import linregress
from auto_texture import generate_FFT
import matplotlib.image as mpimg
import os

def get_vessel_index(file_name):
    # Split the path by '/' and take the last part to get the actual file name
    file_name = file_name.split('/')[-1]
    # Extract the part of the file name that contains the vessel index
    # This assumes the format "translated_axis_kymograph_{index}.png"
    base_name = "central_axis_kymograph_"
    # Removes the prefix and the '.png' part, then converts to integer
    index_part = file_name[len(base_name): -4]  # Removes the specific prefix and ".png"
    return int(index_part)

def est_central_velocity(kymograph_file_path, scale_factor = 3.676, time_factor = 1.667):
    image = mpimg.imread(kymograph_file_path, cv2.IMREAD_GRAYSCALE)
    height, width = np.shape(image)
    _, _, log_magnitude_spectrum = generate_FFT(image)
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

    if np.std(x_filtered) == 0:
        print("Cannot perform linear regression: all x values are identical.")
        velocity_est = 'undetermined'
        return velocity_est

    slope, intercept, r_value, p_value, std_err = linregress(x_filtered, y_filtered)

    

    
    # Use the slope and intercept to calculate the points of the regression line
    regression_line = slope * x_filtered + intercept

    #Use intial estimate of regression line to compute residuals
    res = list(y_filtered - regression_line)
    threshold = 0.5*np.std(res)

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

    print(f"Refined R^2 Value is {np.abs(r_value)}")


    if np.abs(r_value) < 0.30:
        print(f"Orientation of Texture Cannot be Determined")
        velocity_est = 'undetermined'
        return velocity_est

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

    orientation_of_texture = np.arctan(np.abs(slope)) * (180/np.pi) + 90
    orientation_of_texture_rad = np.radians(orientation_of_texture)

    velocity_est = ((width/np.sin(orientation_of_texture_rad)) * scale_factor)/time_factor

    return velocity_est


