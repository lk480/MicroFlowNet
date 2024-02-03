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
    return log_magnitude_spectrum

def low_pass_filter(stiv_array):
    blurred_image = cv2.GaussianBlur(stiv_array, (5, 5), 0)
    normalized_image = np.uint8(255 * blurred_image)
    return normalized_image

def find_dominant_orientation(image, visualize = True):

         
    # Preprocess the image with Gaussian Blur to reduce noise
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

    # Normalize the image
    normalized_image = np.uint8(255 * blurred_image / blurred_image.max())

    # Compute the gradient in x and y direction
    grad_x = cv2.Sobel(normalized_image, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(normalized_image, cv2.CV_64F, 0, 1, ksize=5)

    # Compute the gradient magnitude and orientation for each pixel
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    orientation = np.arctan2(grad_y, grad_x) * (180 / np.pi)
    orientation = np.mod(orientation, 360)  # Use full 360 degrees

    # Construct a histogram of orientations, weighted by gradient magnitude
    hist, bins = np.histogram(orientation, bins=360, range=(0, 360), weights=magnitude)
    hist[0] = 0
    hist[180] = 0

    # Find the predominant orientation
    predominant_orientation = np.argmax(hist)

    print(f"The predominant texture orientation is around {predominant_orientation} degrees.")

    # Visualize the histogram
    if visualize:
        plt.figure(figsize=(10, 6))
        plt.bar(bins[:-1], hist, width=bins[1] - bins[0], color='blue', alpha=0.7)
        plt.title('Histogram of Gradient Orientations')
        plt.xlabel('Orientation (degrees)')
        plt.ylabel('Frequency')
        plt.axvline(x=predominant_orientation, color='red', linestyle='--', label=f'Dominant Orientation: {predominant_orientation}°')
        plt.legend()
        plt.tight_layout()
        plt.show()

    return predominant_orientation

def find_dominant_orientation2(image, visualize=True):
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
    orientation = np.mod(orientation, 360)  # Use full 360 degrees

    # Construct a histogram of orientations, weighted by gradient magnitude
    hist, bins = np.histogram(orientation, bins=360, range=(0, 360), weights=magnitude)
    hist[0] = 0  # Clear the erroneous peak at 0 degrees if necessary
    hist[180] = 0  # Clear the erroneous peak at 180 degrees if necessary

    # Find the predominant orientation
    predominant_orientation = np.argmax(hist)
    print(f"The predominant texture orientation is around {predominant_orientation} degrees.")

    if visualize:
        # Set thresholds
        magnitude_threshold = np.max(magnitude) * 0.5  # 50% of the max value

        # Create a color overlay image
        overlay_image = cv2.cvtColor(normalized_image, cv2.COLOR_GRAY2BGR)

        # Draw the dominant orientation vectors on the image
        step_size = 10  # The step size for the visualization grid
        for i in range(0, image.shape[0], step_size):
            for j in range(0, image.shape[1], step_size):
                if magnitude[i, j] > magnitude_threshold:
                    # Calculate the end point of the vector
                    length = 10  # Length of the vector
                    angle_rad = np.deg2rad(orientation[i, j])
                    end_x = int(j + length * np.cos(angle_rad))
                    end_y = int(i - length * np.sin(angle_rad))  # y is inverted in image coordinates
                    # Draw the line on the overlay image
                    cv2.line(overlay_image, (j, i), (end_x, end_y), (0, 255, 0), 1)

        # Display the overlay image
        plt.figure(figsize=(10, 10))
        plt.imshow(overlay_image)
        plt.title('Gradient Vectors Overlay')
        plt.axis('off')  # Hide the axis
        plt.show()

    return predominant_orientation

# Find and visualize the dominant orientation
dominant_orientation = find_dominant_orientation2(image)

#Spatial FFT
log_magnitude_spectrum = generate_FFT(image)
height, width = np.shape(log_magnitude_spectrum)
log_magnitude_spectrum = cv2.resize(log_magnitude_spectrum, (width, height), interpolation=cv2.INTER_CUBIC)
log_magnitude_spectrum = np.expand_dims(log_magnitude_spectrum, -1)