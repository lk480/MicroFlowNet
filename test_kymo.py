#Generate Translated Axis Kymograph 
import matlab.engine

image_sequence_dir = '/Users/lohithkonathala/Documents/IIB Project/affine_registered_sequences/willeye_affine/'
translated_segment_file_path = '/Users/lohithkonathala/iib_project/translated_vessel_segment.png'

eng = matlab.engine.start_matlab()
binary_image = eng.variable_axis_kymograph_generation(translated_segment_file_path, image_sequence_dir)
eng.quit()

