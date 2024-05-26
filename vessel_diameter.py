import numpy as np
import csv
import matplotlib.pyplot as plt

# Read the CSV file and inspect the column names
with open('/Users/lohithkonathala/Documents/vessel_diameter_axial.csv', mode='r') as file:
    csv_reader = csv.reader(file)
    header = next(csv_reader)
    print("Column Names:", header)

# Read the CSV file
data = np.genfromtxt('/Users/lohithkonathala/Documents/vessel_diameter_axial.csv', delimiter=',', names=True, dtype=None, encoding='utf-8')

# Adjust these based on the actual column names in your CSV file
x_column = header[0]
diameter_column = header[-1]

# Extract columns by name
x_data = data[x_column]  # Adjust the column name if different
diameter_micrometers = data[diameter_column]  # Adjust the column name if different

# Convert diameter to radius in meters
radius_meters = (diameter_micrometers / 2) * 1e-6

# Calculate cross-sectional area (A = πr^2)
cross_sectional_area = np.pi * (radius_meters ** 2)

# Assume a constant flow rate (Q)
flow_rate = 1.0  # m^3/s, this is an arbitrary value for illustration

# Calculate velocity (v = Q / A)
velocity = flow_rate / cross_sectional_area

# Plot the variation in Diameter2
plt.figure(figsize=(10, 6))
plt.plot(x_data, diameter_micrometers, linestyle='-', color='blue', linewidth=2, label='Diameter (micro-meters)')
plt.xlabel('X-Position (Pixels)')
plt.ylabel('Diameter ($\mu m$)')
plt.title('Variation in Diameter along the Vessel')
plt.grid(True, which='both', axis='both', linestyle='--', linewidth=0.7)
plt.show()

# Plot the variation in velocity
plt.figure(figsize=(10, 6))
plt.plot(x_data, velocity, linestyle='-', color='red', linewidth=2, label='Velocity (m/s)')
plt.xlabel('X-Position (Pixels)')
plt.ylabel('Velocity (m/s)')
plt.title('Variation in Velocity along the Vessel')
plt.grid(True, which='both', axis='both', linestyle='--', linewidth=0.7)
plt.show()

