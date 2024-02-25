#Load and Process HVI Data
import cv2
import numpy as np
import os

def register_img(ref, map, method = cv2.MOTION_TRANSLATION):

    # Convert images to grayscale
    im1_gray = ref
    im2_gray = map

    # Find size of image1
    sz = im1_gray.shape

    # Define the motion model
    warp_mode = method

    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    if warp_mode == cv2.MOTION_HOMOGRAPHY :
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else :
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Specify the number of iterations.
    number_of_iterations = 300

    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    termination_eps = 1e-6

    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)

    # Run the ECC algorithm. The results are stored in warp_matrix.
    (cc, warp_matrix) = cv2.findTransformECC (im1_gray,im2_gray,warp_matrix, warp_mode, criteria)

    if warp_mode == cv2.MOTION_HOMOGRAPHY :
    # Use warpPerspective for Homography
        im2_aligned = cv2.warpPerspective (im2_gray, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else :
    # Use warpAffine for Translation, Euclidean and Affine
        im2_aligned = cv2.warpAffine(im2_gray, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    return im1_gray, im2_aligned


# define the function to compute MSE between two images
def mse(img1, img2):
   h = img1.shape[0]
   w = img1.shape[1]
   diff = cv2.subtract(img1, img2)
   err = np.sum(diff**2)
   mse = err/(float(h*w))
   return mse, diff

def find_diff(img1,img2):
    h = img1.shape[0]
    w = img1.shape[1]
    diff = abs(cv2.subtract(img1, img2))
    mse = np.sum(diff**2) / (w*h)
    error = diff/mse
    return error, mse # return the diff picture and mse value


def mii_generator(folder_path, image_end_frame, image_start_frame= 0, img_width = 1944, img_height = 1472):
    image_array = []
    output = np.zeros((img_height,img_width))


    for i in range(image_start_frame,image_end_frame+1):
        file_path = folder_path + f'{i:05d}.pgm'
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
    # print(output.shape)
    output_uint8 = output.astype(np.uint8)
    # print(output_uint8)

    output_file_name = 'MII_start_'+str(image_start_frame)+'_end_'+str(image_end_frame)+'.pgm'
    output_path = folder_path + output_file_name
    cv2.imwrite(output_path, output_uint8)

def create_video_from_images(image_folder, output_video_file, fps=30, frame_size=None):
    """
    Create a video from a sequence of images.

    :param image_folder: Path to the folder containing the images.
    :param output_video_file: Path where the output video will be saved.
    :param fps: Frames per second in the output video.
    :param frame_size: The size of each frame (width, height). If None, the size of the first image is used.
    """
    # Get all image files from the folder, assuming they are sorted in the correct order
    image_files = [f for f in sorted(os.listdir(image_folder)) if f.endswith(('.pgm'))]
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change 'mp4v' to other codecs depending on the format you want
    if not frame_size:
        # Read the first image to set the frame size
        first_image = cv2.imread(os.path.join(image_folder, image_files[0]))
        frame_size = (first_image.shape[1], first_image.shape[0])
    video_writer = cv2.VideoWriter(output_video_file, fourcc, fps, frame_size)

    # Add images to video
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        frame = cv2.imread(image_path)
        if frame_size:
            # Resize frame to match the specified size if necessary
            frame = cv2.resize(frame, frame_size)
        video_writer.write(frame)

    # Release the video writer
    video_writer.release()
    print(f'Video saved to {output_video_file}')

#Image Registration using ECC Algorithm
#Read video
path = '/Users/lohithkonathala/Documents/IIB Project/rigid_body_registered_sequences/test_12x/12x_rigid_body_cropped.mp4'
cap = cv2.VideoCapture(path)

#Get reference frame
ret, ref_frame = cap.read()
if not ret:
    print("Failed to read the frame")
ref_gray = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)
cap.set(cv2.CAP_PROP_POS_FRAMES,1)

cv2.imwrite('/Users/lohithkonathala/Documents/IIB Project/ECC_out/12x_rigid_body_cropped_ecc/00000.pgm', ref_gray)

#Define video output spec
height = ref_frame.shape[0]
width = ref_frame.shape[1]
# print(cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
print(size)

total_mse_original = 0
total_mse_ECC = 0
total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(total_frame)

filler_img = np.zeros((size[1], size[0]), np.uint8)

for i in range(100, 300): # Specify the registration range here.
    ret, frame = cap.read()
    print(f'currently processing image {i:05d}')
    if not ret:
        print("Can't receive frame. Exiting...")
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    try:
        reference, registered = register_img(ref_gray, gray,cv2.MOTION_EUCLIDEAN) # change registration mode here

        error, mse_ECC = find_diff(reference, registered)
        error, mse_original = find_diff(reference, gray)
        total_mse_ECC += mse_ECC
        total_mse_original += mse_original
    
    except:
        registered = gray
        print(f"image number {i:05d} can't be registered, using original instead")

    cv2.imwrite(f'/Users/lohithkonathala/Documents/IIB Project/ECC_out/12x_rigid_body_cropped_ecc/{i:05d}.pgm', registered)


cap.release()
cv2.destroyAllWindows()

image_folder = '/Users/lohithkonathala/Documents/IIB Project/ECC_out/12x_rigid_body_cropped_ecc'
output_video_path = '/Users/lohithkonathala/Documents/IIB Project/ECC_out/12x_rigid_body_cropped_ecc.mp4'
create_video_from_images(image_folder, output_video_path)
