import cv2
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image_path = '/Users/lohithkonathala/iib_project/central_axis_kymograph.png' 
image = mpimg.imread(image_path, cv2.IMREAD_GRAYSCALE)

def generate_FFT(stiv_array):
    fft_image = np.fft.fft2(stiv_array)
    fft_shifted = np.fft.fftshift(fft_image)
    magnitude_spectrum = np.abs(fft_shifted)
    log_magnitude_spectrum = np.log1p(magnitude_spectrum)
    return fft_shifted, magnitude_spectrum, log_magnitude_spectrum

def low_pass_filter(stiv_array):
    blurred_image = cv2.GaussianBlur(stiv_array, (5, 5), 0)
    normalized_image = np.uint8(255 * blurred_image)
    return normalized_image

def historgram_of_gradients(image, visualize=True):
    # Preprocess the image with Gaussian Blur to reduce noise
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

    # Normalize the image
    normalized_image = np.uint8(255 * blurred_image / np.max(blurred_image))

    # Compute the gradient in x and y direction
    grad_x = cv2.Sobel(normalized_image, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(normalized_image, cv2.CV_64F, 0, 1, ksize=5)

    # Compute the gradient magnitude and orientation for each pixel
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    orientation = np.arctan2(grad_y, grad_x) * (180 / np.pi)
    orientation = np.mod(orientation, 360)  

    # Construct a histogram of orientations, weighted by gradient magnitude
    hist, bins = np.histogram(orientation, bins=360, range=(0, 360), weights=magnitude)
    hist[0] = 0  # Clear the erroneous peak at 0 degrees if necessary
    hist[180] = 0  # Clear the erroneous peak at 180 degrees if necessary

    if visualize:
        # Visualization
        plt.figure(figsize=(10, 6))
        plt.bar(bins[:-1], hist, width=bins[1]-bins[0], color='skyblue', edgecolor='black')
        plt.title('Histogram of Orientations Weighted by Gradient Magnitude')
        plt.xlabel('Orientation (Degrees)')
        plt.ylabel('Magnitude Weighted Count')
        plt.xlim(0, 360)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()

    # Find the predominant orientation
    predominant_orientation = np.argmax(hist)
    print(f"The predominant texture orientation is around {predominant_orientation} degrees.")

    return predominant_orientation

def window_function(image):
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

def angular_filter(angle, principal_direction_deg, bandwidth_deg):
    # Convert degrees to radians
    principal_direction_rad = np.deg2rad(principal_direction_deg)
    bandwidth_rad = np.deg2rad(bandwidth_deg)

    # Create the mask
    lower_bound = principal_direction_rad - bandwidth_rad / 2
    upper_bound = principal_direction_rad + bandwidth_rad / 2

    # Handle the wrap-around at 0 and pi
    lower_bound = lower_bound if lower_bound >= 0 else lower_bound + np.pi
    upper_bound = upper_bound if upper_bound <= np.pi else upper_bound - np.pi

    # Create the filter mask
    if lower_bound < upper_bound:
        filter_mask = (angle >= lower_bound) & (angle <= upper_bound)
    else: # Wrap around case
        filter_mask = (angle >= lower_bound) | (angle <= upper_bound)
    
    return filter_mask.astype(int)

#Spatial FFT
windowed_image = window_function(image)
fft_shifted, magnitude_spectrum, log_magnitude_spectrum = generate_FFT(windowed_image)
height, width = np.shape(log_magnitude_spectrum)
log_magnitude_spectrum = cv2.resize(log_magnitude_spectrum, (width, height), interpolation=cv2.INTER_CUBIC)
log_magnitude_spectrum = np.expand_dims(log_magnitude_spectrum, -1)
plt.imshow(log_magnitude_spectrum, cmap='gray')
plt.show()
#Convert Cartesian to Polar
radius, angle = cartesian_to_polar(magnitude_spectrum)

# Calculate |F(θ)|
theta_bins, F_theta = calculate_F_theta(magnitude_spectrum, angle, radius)

# Find the principal direction in the Fourier spectrum
principal_direction = theta_bins[np.argmax(F_theta)] * (180 / np.pi)

# Define the bins for the histogram, which represent the angles in degrees
theta_bins_degrees = np.linspace(0, 180, num=180, endpoint=False)

# Plot the histogram
plt.figure(figsize=(10, 6))
plt.bar(theta_bins_degrees, F_theta, width=1, color='blue', edgecolor='black')
plt.title('Histogram of |F(θ)|')
plt.xlabel('Angle (θ) in Degrees')
plt.ylabel('|F(θ)|')
plt.xlim(0, 180)
plt.grid(True)
plt.show()

# Output the principal direction
print(f"Principal Direction is {principal_direction}")

# Convert to radians and adjust for the FFT's orthogonal representation
principal_direction_rad = np.deg2rad(principal_direction)

# Calculate line coordinates for the overlay
# Assuming the center of the FFT is the origin (0,0)
print(np.shape(log_magnitude_spectrum))
center_y, center_x, _ = np.array(log_magnitude_spectrum.shape) // 2
line_length = np.min([center_x, center_y])  # Line length is half the smallest dimension of the image

# Calculate end points of the line
dx = line_length * np.cos(principal_direction_rad)
dy = line_length * np.sin(principal_direction_rad)
x1, y1 = center_x - dx, center_y - dy
x2, y2 = center_x + dx, center_y + dy

# Plot the FFT image
plt.imshow(log_magnitude_spectrum, cmap='gray')
plt.colorbar()
plt.plot([x1, x2], [y1, y2], 'r', linewidth=2)  # 'r' is the color red
plt.show()

#Apply Angular Filter to FFT
J = angular_filter(angle, principal_direction, bandwidth_deg=20)
plt.imshow(J)
plt.show()

filtered_FFT = fft_shifted * J
filtered_image = np.fft.ifft2(np.fft.ifftshift(filtered_FFT))
filtered_image_real = np.real(filtered_image)
plt.imshow(filtered_image_real, cmap='gray')
plt.show()

#Sanity Check FFT of Inverse 
fft_shifted, magnitude_spectrum, log_magnitude_spectrum = generate_FFT(filtered_image_real)
plt.imshow(log_magnitude_spectrum, cmap='gray')
plt.show()



