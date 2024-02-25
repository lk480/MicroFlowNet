import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import linregress
from scipy import stats
from auto_texture import get_translation_factor, generate_FFT, window_function
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


def generate_velocity_profile(kymograph_dir, visualise=True, verbose=False):
    # Load the image
    files = os.listdir(kymograph_dir)
    sorted_files = sorted(files, key=get_translation_factor)
    print(len(sorted_files))

    #Intialise Lists
    upper_bound_velocities = []
    median_velocities = []
    lower_bound_velocities = []
    translation_factors = []


    for file_name in sorted_files:
        print(f"Translation Factor {get_translation_factor(file_name)}")
        translation_factors.append(get_translation_factor(file_name))
        file_path = os.path.join('/Users/lohithkonathala/iib_project/kymographs', file_name)
        image = mpimg.imread(file_path, cv2.IMREAD_GRAYSCALE)

        #Spatial FFT
        windowed_image = window_function(image)
        fft_shifted, magnitude_spectrum, log_magnitude_spectrum = generate_FFT(image)
        height, width = np.shape(log_magnitude_spectrum)
        center_x = width // 2
        center_y = height // 2
        log_magnitude_spectrum = cv2.resize(log_magnitude_spectrum, (width, height), interpolation=cv2.INTER_CUBIC)

        height, width = np.shape(log_magnitude_spectrum)


        max_val = np.max(log_magnitude_spectrum)
        min_val = np.min(log_magnitude_spectrum)
        scaled_spectrum = ((log_magnitude_spectrum - min_val) / (max_val - min_val)) * 255
        scaled_spectrum = scaled_spectrum.astype(np.uint8)

        _, thresh_image = cv2.threshold(scaled_spectrum, 110, 255, cv2.THRESH_BINARY)

        # Find the coordinates of white pixels
        y, x = np.where(thresh_image == 255)

        # Combine x and y into a single array and sort by x
        points = np.column_stack((x, y))

        if len(points) < 100:
            print('Segment Outside the Vessel')
            print("Velocity Estimate is 0")
            upper_bound_velocities.append(0)
            median_velocities.append(0)
            lower_bound_velocities.append(0)
            continue

        points_sorted = points[np.argsort(points[:, 0])]

        filtered_points_x = points_sorted[points_sorted[:, 0] != center_x]
        filtered_points = filtered_points_x[filtered_points_x[:, 1] != center_y]

        if visualise:
            plt.scatter(filtered_points[:, 0], filtered_points[:, 1], c='red', label='Filtered Points')
            plt.show()

        x_filtered = np.array(filtered_points[:, 0])
        y_filtered = np.array(filtered_points[:, 1])

        thresh1 = 50  # Set your specific lower threshold value for y
        thresh2 = 65  # Set your specific upper threshold value for y

        if visualise:
            # Plot horizontal lines for thresh1 and thresh2
            plt.axhline(y=thresh1, color='blue', linestyle='--', label=f'Thresh1 = {thresh1}')
            plt.axhline(y=thresh2, color='green', linestyle='--', label=f'Thresh2 = {thresh2}')

            plt.scatter(x_filtered, y_filtered, color='blue')

            # Adding labels, title, and legend
            plt.xlabel('X Coordinate')
            plt.ylabel('Y Coordinate')
            plt.title('Filtered Data with Threshold Lines')
            plt.legend()
            plt.gca().set_aspect('equal', adjustable='box')
            plt.show()

        mask = (filtered_points[:, 1] > thresh1) & (filtered_points[:, 1] < thresh2)

        # Apply this mask to filter 'filtered_points'
        filtered_points_within_bounds = filtered_points[mask]

        # If you need to extract x and y arrays separately after this filtering
        x_filtered_within_bounds = filtered_points_within_bounds[:, 0]
        y_filtered_within_bounds = filtered_points_within_bounds[:, 1]

        slope, intercept, r_value, p_value, std_err = linregress(x_filtered_within_bounds, y_filtered_within_bounds)

        print(f"Intial R^2 Value {np.abs(r_value)} and Std Error {std_err}")

        # Use the slope and intercept to calculate the points of the regression line
        regression_line = slope * x_filtered_within_bounds + intercept

        if visualise:
            plt.scatter(x_filtered_within_bounds, y_filtered_within_bounds, color='blue')
            plt.plot(x_filtered_within_bounds, regression_line, color='red', label=f'y = {slope:.2f}x + {intercept:.2f}')
            plt.show()

        #Use intial estimate of regression line to compute residuals
        res = list(y_filtered_within_bounds - regression_line)
        threshold = 0.5*np.std(res)

        data = list(zip(x_filtered_within_bounds, y_filtered_within_bounds))

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

        print(f"Final R^2 Value {np.abs(r_value)} and Std Error {std_err}")

        n = len(x_thresholded_filtered)  # Number of data points
        df = n - 2  # Degrees of freedom for t-distribution

        # For a 95% confidence level, the quantile (for two-tailed test)
        t_score = stats.t.ppf(0.975, df)

        # Confidence interval for the slope
        slope_conf_interval = (slope - t_score * std_err, slope + t_score * std_err)

        # Convert these slope estimates to texture orientation estimates
        max_orientation_estimate = np.arctan(np.abs(slope_conf_interval[0])) * (180/np.pi) + 90
        min_orientation_estimate = np.arctan(np.abs(slope_conf_interval[1])) * (180/np.pi) + 90

        if verbose:
            # Print the results
            print(f"Final R^2 Value: {np.abs(r_value)} and Std Error: {std_err}")
            print(f"95% Confidence Interval for Slope: {slope_conf_interval}")
            print(f"Max Estimate of Main orientation of texture is: {max_orientation_estimate}")
            print(f"Min Estimate of Main orientation of texture is: {min_orientation_estimate}")

        # Use the slope and intercept to calculate the points of the regression line
        regression_line = slope * x_thresholded_filtered + intercept

        if visualise:
            # Plot the filtered data
            plt.scatter(x_thresholded_filtered, y_thresholded_filtered, color='blue')
            plt.plot(x_thresholded_filtered, regression_line, color='red', label=f'y = {slope:.2f}x + {intercept:.2f}')
            plt.xlabel('X-Value')
            plt.ylabel('Y-Value')
            plt.gca().set_aspect('equal', adjustable='box')
            plt.legend()
            # Show the plot
            plt.show()

        orientation_estimate = np.arctan(np.abs(slope)) * (180/np.pi) + 90
        # Print the slope of the line
        print(f"Final Estimate of Main orientation of texture is: {np.arctan(np.abs(slope)) * (180/np.pi) + 90}")

        #Calculate Velocity 
        def get_velocity(N_frames, texture_orientation, scale_factor, time_conv_factor):
            theta = np.radians(texture_orientation - 90)
            distance = (N_frames/np.sin(theta))*scale_factor
            velocity = distance/time_conv_factor
            return velocity

        #Print Velocity Estimates
        upper_bound_velocity = get_velocity(50, min_orientation_estimate, 3.676, 1.667)
        upper_bound_velocities.append(upper_bound_velocity)
        
        median_velocity = get_velocity(50, orientation_estimate, 3.676, 1.667)
        median_velocities.append(median_velocity)
        
        lower_bound_velocity = get_velocity(50, max_orientation_estimate, 3.676, 1.667)
        lower_bound_velocities.append(lower_bound_velocity)


    #Visualise Profile
    errors = [[value - lower, upper - value] for lower, value, upper in zip(lower_bound_velocities, median_velocities, upper_bound_velocities)]
    lower_errors = [value - lower for lower, value in zip(lower_bound_velocities, median_velocities)]
    upper_errors = [upper - value for upper, value in zip(upper_bound_velocities, median_velocities)]
    asymmetric_error = [lower_errors, upper_errors]

    # Create the plot
    plt.figure()
    plt.errorbar(translation_factors, median_velocities, yerr=asymmetric_error, capsize=5, capthick=2, ecolor='red', marker='s', markersize=5, linestyle='--', linewidth=2)

    # Customize the plot
    plt.title('Velocity Profile with Upper and Lower Bounds (95% Confidence Interval)')
    plt.xlabel('Translation Factor')
    plt.ylabel('Velocity')

    plt.savefig('velocity_profile.png', bbox_inches='tight')
    # Show the plot
    plt.show()
