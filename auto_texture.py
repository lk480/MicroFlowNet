import cv2
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

def generate_FFT(stiv_array):
    """
    Args:
        stiv_array (arr): array containing space-time image (i.e. position on y-axis and time on x-axis - a.k.a kymograph)

    Returns:
        tuple: fft_shifted, magnitude_spectrum, log_magnitude_spectrum
    """
    fft_image = np.fft.fft2(stiv_array)
    fft_shifted = np.fft.fftshift(fft_image)
    magnitude_spectrum = np.abs(fft_shifted)         
    log_magnitude_spectrum = np.log1p(magnitude_spectrum)
    return fft_shifted, magnitude_spectrum, log_magnitude_spectrum

def low_pass_filter(stiv_array):
    """
    Args:
        stiv_array (arr): array containing space-time image (i.e. position on y-axis and time on x-axis - a.k.a kymograph)

    Returns:
        arr: low-pass filtered space-time image
    """
    blurred_image = cv2.GaussianBlur(stiv_array, (5, 5), 0)
    normalized_image = np.uint8(255 * blurred_image)
    return normalized_image

def histogram_of_gradients(image, visualize=True):
    """Function to compute histogram of gradients for a given space-time image followed by selection of the pre-dominant texture

    Args:
        image (arr): image stored as a NumPy array  
        visualize (bool, optional): _description_. Defaults to True.

    Returns:
        tuple: predominant_orientation, probability_distribution
    """

    # Preprocess the image with Gaussian Blur to reduce noise
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    # Normalize the image
    normalized_image = np.uint8(255 * blurred_image / np.max(blurred_image))
    # Compute the gradient in x and y direction
    grad_x = cv2.Sobel(normalized_image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(normalized_image, cv2.CV_64F, 0, 1, ksize=3)

    # Compute the gradient magnitude and orientation for each pixel
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    orientation = np.arctan2(grad_y, grad_x) * (180 / np.pi)
    orientation = np.mod(orientation, 180)  

    # Construct a histogram of orientations, weighted by gradient magnitude
    hist, bins = np.histogram(orientation, bins=360, range=(0, 180), weights=magnitude)
    hist[0] = 0
    hist[90] = 0
    hist[180] = 0
    
    if visualize:
        # Visualization
        plt.figure(figsize=(10, 6))
        plt.bar(bins[:-1], hist, width=bins[1]-bins[0], color='skyblue', edgecolor='black')
        plt.title('Histogram of Orientations Weighted by Gradient Magnitude')
        plt.xlabel('Orientation (Degrees)')
        plt.ylabel('Magnitude Weighted Count')
        plt.xlim(0, 180)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()

    # Find the predominant orientation
    max_index = np.argmax(hist)
    predominant_orientation = (bins[max_index] + bins[max_index + 1])/2
    print(f"The predominant texture orientation is around {predominant_orientation} degrees.")


    #Find Probability Distribution
    prob_dist = hist/np.sum(hist)
    #print(f"Sum of probability distribution {np.sum(prob_dist)}")
    print(np.max(prob_dist), np.min(prob_dist))

    return predominant_orientation, prob_dist

def window_function(image):
    """Function to apply a Hanning window to the image

    Args:
        image (arr): image stored as NumPy array

    Returns:
        arr: image with window function applied
    """
    M, N = np.shape(image)
    W_m = 0.5 * (1 - np.cos(2 * np.pi * np.arange(M) / M))
    W_n = 0.5 * (1 - np.cos(2 * np.pi * np.arange(N) / N))
    W = np.outer(W_m, W_n)
    windowed_image = W * image
    return windowed_image

def cartesian_to_polar(fft_img):
    # Get the center of the image
    center_x, center_y = np.array(fft_img.shape) // 2
    # Create an array with the size of the FFT image, filled with the radius values
    Y, X = np.ogrid[:fft_img.shape[0], :fft_img.shape[1]]
    r = np.hypot(X - center_x, Y - center_y)
    # Create an array with the size of the FFT image, filled with the angle values
    theta = np.arctan2(Y - center_y, X - center_x)
    # Adjust theta to be between 0 and pi (0 and 180 degrees)
    theta[theta < 0] += np.pi
    return r, theta

# Function to calculate |F(θ)|
def calculate_F_theta(magnitude_spectrum, angle, radius):
    # Define the number of bins for the histogram
    theta_bins = np.linspace(0, np.pi, num=180, endpoint=False)
    # Initialize |F(θ)| to be all zeros
    F_theta = np.zeros_like(theta_bins)
    
    # Calculate |F(θ)| by summing the magnitudes for each θ
    for i, theta in enumerate(theta_bins):
        # Find the pixels that have angles within the range of the current bin
        mask = (angle >= theta) & (angle < theta + np.pi/180)
        # Integrate the magnitude spectrum within the mask, this is |F(θ)|
        F_theta[i] = np.sum(magnitude_spectrum[mask])
    
    return theta_bins, F_theta

def angular_filter(log_magnitude_spectrum, lower_bound, upper_bound):
    #Apply Angular Filter to FFT
    rows, cols, _ = np.shape(log_magnitude_spectrum)
    crow, ccol = rows // 2, cols // 2  # Center

    # Create an array that holds the angles of each pixel relative to the center
    y, x = np.ogrid[-crow:rows-crow, -ccol:cols-ccol]
    angle = np.rad2deg(np.arctan2(y, x)) % 360.0  # Convert from radians to degrees and normalize to [0, 360)

    lower_bound = lower_bound % 360
    upper_bound = upper_bound % 360
    shifted_lower_bound = (lower_bound + 180) % 360
    shifted_upper_bound = (upper_bound + 180) % 360

    # Create the original filter mask
    if lower_bound < upper_bound:
        original_filter_mask = (angle >= lower_bound) & (angle <= upper_bound)
    else:  # Wrap around case
        original_filter_mask = (angle >= lower_bound) | (angle <= upper_bound)

    # Create the shifted filter mask
    if shifted_lower_bound < shifted_upper_bound:
        shifted_filter_mask = (angle >= shifted_lower_bound) & (angle <= shifted_upper_bound)
    else:  # Wrap around case
        shifted_filter_mask = (angle >= shifted_lower_bound) | (angle <= shifted_upper_bound)
    # Combine both masks
    filter_mask = original_filter_mask | shifted_filter_mask

    return filter_mask

# Function to extract the translation factor from the file name
def get_translation_factor(file_name):
    # Extract the part of the file name that contains the translation factor
    factor_part = file_name.split('_')[-1]  # Splits by underscore and takes the last part
    # Convert to float
    try:
        return float(factor_part[:-4])  # Removes the last 4 characters (e.g., ".png") and converts to float
    except ValueError:
        return 0.0  # Default value in case of any conversion error

def convert_translation_factor(translation_factor, scale_factor):
    offset = translation_factor * scale_factor
    return offset




