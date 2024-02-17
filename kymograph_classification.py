from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, LeakyReLU, Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras_cv.layers import DropBlock2D
from keras.models import Model

def kymograph_CNN(input_size=(256,256,1), drop_rate=0.2, kernel_init='he_normal'):
    inputs = Input(input_size)
    conv1 = Conv2D(32, kernel_size=(7,7), activation=LeakyReLU(alpha=0.3), kernel_initializer=kernel_init)(inputs)
    conv2 = Conv2D(32, kernel_size=(5,5), activation=LeakyReLU(alpha=0.3), kernel_initializer=kernel_init)(conv1)
    pool1 = MaxPooling2D((2,2))(conv2)
    dropblock1 = Dropout(rate=drop_rate)(pool1)

    conv3 = Conv2D(64, kernel_size=(5,5), activation=LeakyReLU(alpha=0.3), kernel_initializer=kernel_init)(dropblock1)
    conv4 = Conv2D(64, kernel_size=(3,3), activation=LeakyReLU(alpha=0.3), kernel_initializer=kernel_init)(conv3)
    pool2 = MaxPooling2D((2,2))(conv4)
    dropblock2 = DropBlock2D(rate=drop_rate)(pool2)

    conv5 = Conv2D(128, kernel_size=(5,5), activation=LeakyReLU(alpha=0.3), kernel_initializer=kernel_init)(dropblock2)
    conv6 = Conv2D(128, kernel_size=(3,3), activation=LeakyReLU(alpha=0.3), kernel_initializer=kernel_init)(conv5)
    pool3 = MaxPooling2D((2,2))(conv6)
    dropblock3 = DropBlock2D(rate=drop_rate)(pool3)

    conv7 = Conv2D(512, kernel_size=(5,5), activation=LeakyReLU(alpha=0.3), kernel_initializer=kernel_init)(dropblock3)
    conv8 = Conv2D(512, kernel_size=(3,3), activation=LeakyReLU(alpha=0.3), kernel_initializer=kernel_init)(conv7)
    global_avg_pool = GlobalAveragePooling2D()(conv8)

    flatten = Flatten()(global_avg_pool)
    out_layer = Dense(180, activation ='softmax', kernel_initializer=kernel_init)(flatten)
    
    model = Model(inputs=[inputs], outputs=[out_layer])
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    return model 



























