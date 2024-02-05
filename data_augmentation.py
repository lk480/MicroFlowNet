import numpy as np

def augment_i(x, y, size=(64,64)):
    dim0 = x.shape[0]
    dim1 = x.shape[1]
    dim2 = x.shape[2]
    #get image index
    image_index = np.random.randint(0, dim0)
    min_h = np.random.randint(0, x.shape[1]-size[0])
    max_h = min_h+size[0]
    min_w = np.random.randint(0, x.shape[2]-size[1])
    max_w = min_w+size[1]
    # extract snippet
    im_x = x[image_index, min_h:max_h, min_w:max_w, :]
    im_y = y[image_index, min_h:max_h, min_w:max_w, :]

    # rotate
    number_rotaions = np.random.randint(0,4)
    im_x = np.rot90(im_x, k=number_rotaions, axes=(0,1))
    im_y = np.rot90(im_y, k=number_rotaions, axes=(0,1))
        
    # flip left-right, up-down
    if np.random.random() < 0.5:
        lr_ud = np.random.randint(0,2) # flip up-down or left-right?
        im_x = np.flip(im_x, axis=lr_ud)
        im_y = np.flip(im_y, axis=lr_ud)
            
    return (im_x, im_y)

def augment_data(x, y, N, size):
    augmented_data=[]
    for i in range(N):
        augmented_data.append(augment_i(x, y, size))
     
    x_augment = np.array([i[0] for i in augmented_data])
    y_augment = np.array([i[1] for i in augmented_data])
    return (x_augment, y_augment)
