import os
import cv2

"""Pre-process the frames using a combination of CLAHE and Fast Non-Local Means Denoising"""

#Specify Input Sequence Directory
image_sequence_dir = '/Users/lohithkonathala/Documents/IIB Project/12x_rigid_body_final'
#Specify Output Sequence Directory
output_sequence_dir = '/Users/lohithkonathala/Documents/IIB Project/12x_rigid_body_final_denoised'

# Create the output directory if it doesn't exist
if not os.path.exists(output_sequence_dir):
    os.makedirs(output_sequence_dir)

# Intialise Contrasted Limited Adaptive Histogram Equalization 
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))

for image_file_name in os.listdir(image_sequence_dir):
    file_path = os.path.join(image_sequence_dir, image_file_name)
    output_file_path = os.path.join(output_sequence_dir, image_file_name)  # Output path with the same file name
    
    # Read the image in grayscale mode
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        continue  # Skip files that aren't images

    # Apply CLAHE
    image_clahe = clahe.apply(image)
    
    # Apply Non-Local Means Denoising
    denoised_image = cv2.fastNlMeansDenoising(image_clahe, None, 15, 7, 21)
    
    # Save the processed image to the output directory
    cv2.imwrite(output_file_path, denoised_image)

print("Processing complete. Images have been saved to:", output_sequence_dir)