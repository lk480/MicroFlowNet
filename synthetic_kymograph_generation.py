import numpy as np
from PIL import Image
from perlin_noise import PerlinNoise
from perlin_numpy import generate_fractal_noise_2d
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
import os 
import cv2
from auto_texture import get_translation_factor, generate_FFT, window_function, histogram_of_gradients
from tqdm import tqdm 

def generate_directional_perlin(width, height, angle_degrees, scale, num_layers, octaves, persistence, lacunarity):
    noise = PerlinNoise()
    angle_radians = np.radians(angle_degrees)

    def perlin(x, y):
        # Generate Perlin noise value at a given (x, y) position
        return noise([x * np.cos(angle_radians) + y * np.sin(angle_radians)])

    shape = (height, width)
    world = np.zeros(shape)

    for i in range(height):
        for j in range(width):
            x = i / height * scale
            y = j / width * scale
            value = 0.0
            for layer in range(num_layers):
                layer_scale = scale / (2 ** layer)
                value += perlin(x/layer_scale, y/layer_scale) / (2 ** layer)

            value = (value + 1.0) / 2.0  # Normalize to [0, 1]
            amplitude = 0.01
            frequency = 0.01

            for _ in range(octaves):
              value += noise([x * frequency, y * frequency]) * amplitude
              amplitude *= persistence
              frequency *= lacunarity
            world[i][j] = value

    world = (world - np.min(world)) / (np.max(world) - np.min(world))  # Normalize to [0, 1]
    return world


def generate_base_image():
    # Generate each stripe pattern with Perlin noise, now including lacunarity
    noise = generate_fractal_noise_2d(shape=(256, 256), res=(8, 8), octaves=5, persistence=0.8)
    noise_arr = np.array(noise)
    return noise_arr

def generate_image(noise_array):
    height, width = noise_array.shape
    image = Image.new('L', (width, height))

    for y in range(height):
        for x in range(width):
            value = noise_array[y][x]
            color_value = int(255 * value)
            image.putpixel((x, y), color_value)
    image = np.array(image)/255.0
    return image

def sub_sample(sample_image):
    sub_sampled_image = sample_image[::1, ::1]
    # Add random noise to the sub-sampled image
    noise = np.random.normal(0, 0.05, sub_sampled_image.shape)
    noisy_image = sub_sampled_image + noise
    noisy_image = np.clip(noisy_image, 0, 1)
    return noisy_image

def crop_center(img, new_height, new_width):
    """Crop the center of the image to the desired size."""
    height, width = img.shape[:2]
    startx = width//2 - new_width//2
    starty = height//2 - new_height//2    
    return img[starty:starty+new_height, startx:startx+new_width]


orientations_list = list(np.arange(165, 180, 0.5))
main_directory = '/Users/lohithkonathala/iib_project/synthetic_kymographs/'

for orientation in orientations_list:
    # Create a subdirectory for each orientation
    print(f"Processing Angle of Orientation: {orientation}")
    sub_directory = os.path.join(main_directory, f"Angle_{orientation}")
    if not os.path.exists(sub_directory):
        os.makedirs(sub_directory)

    for i in tqdm(range(100), desc=f"Generating for angle {orientation}"):  
        # Assuming these functions generate the images you mentioned
        directional_perlin_noise = generate_directional_perlin(width=128, height=128, angle_degrees=orientation, scale=1, num_layers=15, octaves=10, persistence=0.1, lacunarity=2.5)
        base_noise = generate_base_image()
        rotated_base_noise = rotate(base_noise, orientation, reshape=False)
        resized_base_noise = crop_center(rotated_base_noise, 128, 128)
        weighted_sum = 0.9 * directional_perlin_noise + 0.1 * resized_base_noise
        weighted_sum_normalized = (weighted_sum - np.min(weighted_sum)) / (np.max(weighted_sum) - np.min(weighted_sum))
        weighted_sum_normalized_uint8 = (255 * weighted_sum_normalized).astype(np.uint8) 
        
        # Save weighted_sum_normalized image
        normalized_filename = os.path.join(sub_directory, f"{orientation}_instance_{i+1}.png")
        cv2.imwrite(normalized_filename, weighted_sum_normalized_uint8)

        _, _, log_magnitude_spectrum = generate_FFT(weighted_sum_normalized)
        log_magnitude_spectrum_uint8 = (255 * log_magnitude_spectrum / np.max(log_magnitude_spectrum)).astype(np.uint8)
        fft_filename = os.path.join(sub_directory, f"log_magnitude_{orientation}_instance_{i+1}.png")
        cv2.imwrite(fft_filename, log_magnitude_spectrum_uint8)

        









