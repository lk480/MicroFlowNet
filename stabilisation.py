import os
import cv2
from tqdm import tqdm
from pystackreg import StackReg
from skimage import io
import numpy as np

def stabilise_sequence(image_sequence_folder, img_frame_start, img_frame_end, 
                       frame_rate, video_path, stabilised_frames_folder, 
                       transformation_mode=StackReg.AFFINE):
    """
    Stabilizes a sequence of images using the specified transformation mode.

    Args:
        image_sequence_folder (str): Folder containing the image sequence.
        img_frame_start (int): Index of the first frame to include.
        img_frame_end (int): Index of the last frame to include.
        frame_rate (float): Frame rate for the output video.
        video_path (str): Path to save the output video.
        stabilised_frames_folder (str): Folder to save the stabilized frames.
        transformation_mode (int): Transformation mode for StackReg. Default is StackReg.AFFINE.
    """
    if not os.path.exists(stabilised_frames_folder):
        os.makedirs(stabilised_frames_folder)

    # Sort and filter images based on provided frame indices
    images = sorted(os.listdir(image_sequence_folder))
    images = [img for img in images if img.lower().endswith('.pgm') and not img.startswith('.')]
    images = images[img_frame_start:img_frame_end]

    # Create a StackReg object for the specified transformation
    sr = StackReg(transformation_mode)

    # Load the first image to set video properties
    first_frame = io.imread(os.path.join(image_sequence_folder, images[0]))
    height, width = first_frame.shape[:2]

    # Define the codec and create VideoWriter object
    video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'MJPG'), frame_rate, (width, height))

    # Perform the registration and write to video and files
    for idx, image_name in enumerate(tqdm(images)):
        image_path = os.path.join(image_sequence_folder, image_name)
        image = io.imread(image_path)

        # Use the first image as the reference
        if idx == 0:
            ref_image = image

        # Perform registration
        transformed_image = sr.register_transform(ref_image, image)

        # Convert the transformed image to 8-bit and then to BGR color space for video writing
        transformed_image_8bit = np.clip(transformed_image, 0, 255).astype('uint8')
        transformed_image_bgr = cv2.cvtColor(transformed_image_8bit, cv2.COLOR_RGB2BGR)

        # Write the transformed image to video
        video.write(transformed_image_bgr)

        # Save the transformed image as a file in the specified folder
        frame_filename = os.path.join(stabilised_frames_folder, f"frame_{idx:04d}.pgm")
        io.imsave(frame_filename, transformed_image_8bit)

    # Release video writer and destroy all windows
    video.release()
    cv2.destroyAllWindows()

def crop_frame(x_topleft, y_topleft, width, height, input_folder, output_folder):
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # List all files in the input folder and sort them if necessary
    files = sorted([f for f in os.listdir(input_folder) if f.endswith('.pgm')])  

    # Crop each image in the sequence
    for file in files:
        # Read the image
        img = cv2.imread(os.path.join(input_folder, file), cv2.IMREAD_UNCHANGED)  # Read the image in its original format (unchanged)

        # Validate if the image has been read correctly
        if img is None:
            print(f"Error reading image {file}")
            continue

        # Crop the image
        cropped_img = img[y_topleft:y_topleft+height, x_topleft:x_topleft+width]

        # Convert to grayscale if not already (important for PGM format)
        if len(cropped_img.shape) == 3:  # Indicates that the image is not grayscale
            cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)

        # Save the cropped image to the output folder
        cv2.imwrite(os.path.join(output_folder, file), cropped_img)

    print("Cropping complete!")


def mii_generator(folder_path, image_end_frame, image_start_frame, img_width, img_height):
    image_array = []
    output = np.zeros((img_height,img_width))

    for i in range(image_start_frame,image_end_frame+1):
        file_path = os.path.join(folder_path, f"frame_{i:04d}.pgm")
        print(file_path)
        frame = cv2.imread(file_path)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #image_array[i] = gray
        image_array.append(gray)

    # image_array = image_array.astype(int)
    # print(image_array[1])
    output = image_array[0]

    for i in range(image_end_frame - image_start_frame + 1):
        output = np.fmin(output, image_array[i])

    output = output.astype(int)
    output_uint8 = output.astype(np.uint8)

    output_file_name = 'MII_start_'+str(image_start_frame)+'_end_'+str(image_end_frame)+'.pgm'
    output_path = os.path.join(folder_path, output_file_name)
    cv2.imwrite(output_path, output_uint8)


#Specify Directories
image_sequence_folder =  '/Users/lohithkonathala/Documents/IIB Project/raw_hvi_sequences.noysnc/V4HYP1001LT0-220'
video_path = '/Users/lohithkonathala/Documents/IIB Project/12x_rigid_body.mp4'
stabilised_frames_folder = '/Users/lohithkonathala/Documents/IIB Project/12x_rigid_body'

#Specify Sequence Properties
img_frame_start = 0
img_frame_end = 50
frame_rate = 30

#Apply Affine Stabilisation to Test Sequence 
#stabilise_sequence(image_sequence_folder, img_frame_start, img_frame_end, frame_rate, video_path, stabilised_frames_folder, transformation_mode=StackReg.AFFINE)

#Crop Image
x_topleft = 775
y_topleft = 300
width = 512
height = 512

input_folder = '/Users/lohithkonathala/Documents/IIB Project/12x_rigid_body'
output_folder = '/Users/lohithkonathala/Documents/IIB Project/12x_rigid_body_cropped'

#crop_frame(x_topleft, y_topleft, width, height, input_folder, output_folder)

#Apply Rigid Body Stabilisation to Cropped Sequence
image_sequence_folder =  '/Users/lohithkonathala/Documents/IIB Project/12x_rigid_body_cropped'
video_path = '/Users/lohithkonathala/Documents/IIB Project/12x_rigid_body_final.mp4'
stabilised_frames_folder = '/Users/lohithkonathala/Documents/IIB Project/12x_rigid_body_final'

#Specify Sequence Properties
img_frame_start = 0
img_frame_end = 50
frame_rate = 30

#stabilise_sequence(image_sequence_folder, img_frame_start, img_frame_end, frame_rate, video_path, stabilised_frames_folder, transformation_mode=StackReg.RIGID_BODY)

#Generate MII 
folder_path = '/Users/lohithkonathala/Documents/IIB Project/12x_affine'
mii_generator(folder_path, image_end_frame=9, image_start_frame=0, img_width=1360, img_height=1024)