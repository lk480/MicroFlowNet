import cv2
import numpy as np
import matplotlib.pyplot as plt
from data_augmentation import augment_data
from segmentation_model import load_hvi_image
from keras.optimizers import Adam
from segmentation_model import SA_UNet
import tensorflow as tf
from keras.callbacks import ReduceLROnPlateau


#File Paths/Directories
pretrained_weight_filepath = '/Users/lohithkonathala/iib_project/sa_unet_CHASE_weights.h5'
fine_tuned_weight_filepath =  '/Users/lohithkonathala/iib_project/sa_unet_tuned_weights.h5'

def model_fine_tune(training_data, validation_data, pretrained_weight_filepath, fine_tuned_weight_filepath):
    model = SA_UNet()
    model.load_weights(pretrained_weight_filepath)

    for layer in model.layers[:-3]:
        layer.trainable = False
    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=0.00001, verbose=1)
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(training_data, epochs=10, callbacks=[reduce_lr], validation_data=validation_data)
    model.save(fine_tuned_weight_filepath)


#Load HVI and Ground Truths
x1 = load_hvi_image('/Users/lohithkonathala/Documents/IIB Project/HVI_Manual_Annotations/jed_eye_affine_MII.pgm')
x2 = load_hvi_image('/Users/lohithkonathala/Documents/IIB Project/HVI_Manual_Annotations/lohith_eye_affine_MII.pgm')
x_data = np.array([x1.numpy(), x2.numpy()])
x_data = x_data.astype('float32')/1.0

y1 = cv2.imread('/Users/lohithkonathala/iib_project/hvi_ground_truths/jed_eye_affine_mii_gt.png')
y2 = cv2.imread('/Users/lohithkonathala/iib_project/hvi_ground_truths/lohith_eye_affine_mii_gt.png')
y1 = np.expand_dims(cv2.cvtColor(y1, cv2.COLOR_BGR2GRAY), -1)
y2 = np.expand_dims(cv2.cvtColor(y2, cv2.COLOR_BGR2GRAY), -1)
y_data = np.array([y1, y2])
y_data = y_data.astype('float32')/255


X_aug, Y_aug = augment_data(x_data, y_data, N=2000, size=(512,512))

# Create Training Dataset
dataset = tf.data.Dataset.from_tensor_slices((X_aug, Y_aug))

# Shuffle, batch, and prefetch the dataset
BUFFER_SIZE = 1000
BATCH_SIZE = 32
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

# Calculate the number of batches in train and validation set
total_size = len(list(dataset))
train_size = int(0.8 * total_size)
val_size = total_size - train_size

# Split the dataset
train_data = dataset.take(train_size)
val_data = dataset.skip(train_size)

# Configure the datasets for performance
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_data.cache().prefetch(buffer_size=AUTOTUNE)
val_dataset = val_data.cache().prefetch(buffer_size=AUTOTUNE)

# Fine Tune Model (assuming you've updated the model_fine_tune function to accept validation data)
model_fine_tune(train_dataset, val_dataset, pretrained_weight_filepath, fine_tuned_weight_filepath)






