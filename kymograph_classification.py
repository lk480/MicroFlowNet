from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, LeakyReLU, Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras_cv.layers import DropBlock2D
from keras.models import Model
import os
import tensorflow as tf
import random

def kymograph_CNN(input_size=(128,128,1), drop_rate=0.2, kernel_init='he_normal'):
    inputs = Input(input_size)
    conv1 = Conv2D(32, kernel_size=(7,7), activation=LeakyReLU(alpha=0.3), kernel_initializer=kernel_init)(inputs)
    conv2 = Conv2D(32, kernel_size=(5,5), activation=LeakyReLU(alpha=0.3), kernel_initializer=kernel_init)(conv1)
    pool1 = MaxPooling2D((2,2))(conv2)
    dropblock1 = Dropout(rate=drop_rate)(pool1)

    conv3 = Conv2D(64, kernel_size=(5,5), activation=LeakyReLU(alpha=0.3), kernel_initializer=kernel_init)(dropblock1)
    conv4 = Conv2D(64, kernel_size=(3,3), activation=LeakyReLU(alpha=0.3), kernel_initializer=kernel_init)(conv3)
    pool2 = MaxPooling2D((2,2))(conv4)
    dropblock2 = Dropout(rate=drop_rate)(pool2)

    conv5 = Conv2D(128, kernel_size=(5,5), activation=LeakyReLU(alpha=0.3), kernel_initializer=kernel_init)(dropblock2)
    conv6 = Conv2D(128, kernel_size=(3,3), activation=LeakyReLU(alpha=0.3), kernel_initializer=kernel_init)(conv5)
    pool3 = MaxPooling2D((2,2))(conv6)
    dropblock3 = Dropout(rate=drop_rate)(pool3)

    conv7 = Conv2D(512, kernel_size=(5,5), activation=LeakyReLU(alpha=0.3), kernel_initializer=kernel_init)(dropblock3)
    conv8 = Conv2D(512, kernel_size=(3,3), activation=LeakyReLU(alpha=0.3), kernel_initializer=kernel_init)(conv7)
    global_avg_pool = GlobalAveragePooling2D()(conv8)

    flatten = Flatten()(global_avg_pool)
    out_layer = Dense(180, activation ='softmax', kernel_initializer=kernel_init)(flatten)
    
    model = Model(inputs=[inputs], outputs=[out_layer])
    model.compile(optimizer=Adam(learning_rate=0.01), loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return model 

def load_fft_paths(synthetic_kymograph_dir):
    fft_files = []  # Initialize an empty list to store the paths of matching files
    for dir_name in os.listdir(synthetic_kymograph_dir):
        # Construct the full path to ensure we are checking a directory
        dir_path = os.path.join(synthetic_kymograph_dir, dir_name)
        # Check if this path is indeed a directory
        if os.path.isdir(dir_path):
            files = os.listdir(dir_path)
            fft_files.extend(os.path.join(dir_path, f) for f in files if f.startswith('log_magnitude'))
    return fft_files 
        
def extract_labels(file_paths):
    labels = set()  # Using a set to avoid duplicate values
    for file_path in file_paths:
        # Extract the filename from the full path and split it into parts
        filename = os.path.basename(file_path)
        parts = filename.split('_')  # Splitting by underscore
        if len(parts) > 2:  # Check if the filename format is correct
            label = parts[2]  # Get the term that is supposed to be the numerical value
            labels.add(label)  # Add the extracted label to the set
    return labels

def load_data(file_path, size=(128,128)):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_png(image, 1)
    image = tf.image.resize(image, size)
    image = tf.cast(image, tf.float32) / 255.0
    return image

def create_train_test_val(synthetic_kymograph_dir):
    fft_files = load_fft_paths(synthetic_kymograph_dir)
    #Create training, test and validation datasets
    random.shuffle(fft_files) 
    # Define split proportions
    train_split = 0.70  
    val_split = 0.15   
    test_split = 0.15  

    # Calculate split indices
    num_files = len(fft_files)
    train_end = int(train_split * num_files)
    val_end = train_end + int(val_split * num_files)

    # Split the data
    train_files = fft_files[:train_end]
    val_files = fft_files[train_end:val_end]
    test_files = fft_files[val_end:]
    
    return train_files, val_files, test_files

def one_hot_encoding(fft_files):
    labels_asc = sorted(extract_labels(fft_files), key=float)
    label_to_index = {label: idx for idx, label in enumerate(labels_asc)}
    label_indices = [label_to_index[label] for label in labels_asc]
    one_hot_labels = tf.one_hot(indices=label_indices, depth=180)
    return one_hot_labels

train_file_paths, val_file_paths, test_file_paths = create_train_test_val('/Users/lohithkonathala/iib_project/synthetic_kymographs')

train_labels_enc = one_hot_encoding(train_file_paths)
train_labels_tensor = tf.data.Dataset.from_tensor_slices(train_labels_enc)
val_labels_enc = one_hot_encoding(val_file_paths)
val_labels_tensor = tf.data.Dataset.from_tensor_slices(val_labels_enc)
test_labels_enc = one_hot_encoding(test_file_paths)
test_labels_tensor = tf.data.Dataset.from_tensor_slices(test_labels_enc)

train_data_tensor = tf.data.Dataset.from_tensor_slices(train_file_paths)
train_data_tensor = train_data_tensor.map(load_data)
val_data_tensor = tf.data.Dataset.from_tensor_slices(val_file_paths)
val_data_tensor = val_data_tensor.map(load_data)
test_data_tensor = tf.data.Dataset.from_tensor_slices(test_file_paths)
test_data_tensor = test_data_tensor.map(load_data)

train_dataset = tf.data.Dataset.zip(train_data_tensor, train_labels_tensor)
validation_dataset = tf.data.Dataset.zip(val_data_tensor, val_labels_tensor)
test_dataset = tf.data.Dataset.zip(test_data_tensor, test_data_tensor)

training_data = train_dataset.batch(batch_size=64)
validation_data = validation_dataset.batch(batch_size=64)

#Load and Compile CNN
model = kymograph_CNN()
model.fit(
    training_data,  # Your training data
    epochs=50,  # Number of epochs to train for
    validation_data=validation_data  # Your validation data
)