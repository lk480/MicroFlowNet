import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, fftshift

def combined_sinusoidal_noise_single_angle(x, y, num_components=5, base_frequency=5, amplitude=1, angle=np.pi/4):
    noise = np.zeros_like(x)
    for i in range(num_components):
        frequency = base_frequency * (i + 1)  # Increase frequency for each component
        u_0 = frequency * np.cos(angle)
        v_0 = frequency * np.sin(angle)
        noise += amplitude * np.sin(2 * np.pi * (u_0 * x + v_0 * y))
    return noise

# Parameters
size = 256
base_frequency = 5
amplitude = 1
angle = np.pi / 4  # 45 degrees

# Create a grid of points
x = np.linspace(0, 10, size)
y = np.linspace(0, 10, size)
x, y = np.meshgrid(x, y)

# Generate combined sinusoidal noise with a single angle
noise = combined_sinusoidal_noise_single_angle(x, y, num_components=5, base_frequency=base_frequency, amplitude=amplitude, angle=angle)

# Compute the 2D Fourier Transform
F_noise = fft2(noise)
F_noise_shifted = fftshift(F_noise)  # Shift the zero frequency component to the center

# Compute the magnitude spectrum
magnitude_spectrum = np.abs(F_noise_shifted)

# Plot the original noise
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(noise, cmap='gray', extent=(0, 10, 0, 10))
plt.title('Sinusoidal Noise')

# Plot the magnitude spectrum
plt.subplot(1, 2, 2)
plt.imshow(np.log(1 + magnitude_spectrum), cmap='gray', extent=(-size//2, size//2, -size//2, size//2))
plt.title('Magnitude Spectrum')

# Overlay line connecting points of elevated intensity
x_line = np.linspace(-size//2, size//2, 1000)
y_line = (-x_line * np.tan(angle))  # Line at the given angle, corrected direction
plt.plot(x_line, y_line, 'w--', linewidth=1)  # Dashed white line

plt.show()
