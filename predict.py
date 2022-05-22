import tensorflow as tf
import numpy as np
from PIL import Image


def unpadding(lr_image, patches_per_row, patches_per_col, patch_size, padded_image):
    # Get the dimensions of the low resolution image
    _, lr_rows, lr_cols, _ = lr_image.shape

    # the number of padded rows and columns
    padded_rows = patches_per_row * patch_size
    padded_cols = patches_per_col * patch_size

    row_padding = padded_rows - lr_rows
    col_padding = padded_cols - lr_cols

    # Get the scaling factor
    scaling_factor = padded_image.shape[0] // padded_rows

    extract_from_row = int(row_padding / 2 * scaling_factor)
    extract_till_row = -1

    if row_padding > 0:
        extract_till_row = - extract_from_row

    extract_from_col = int(col_padding / 2 * scaling_factor)
    extract_till_col = -1

    if col_padding > 0:
        extract_till_col = - extract_from_col

    return padded_image[extract_from_row:extract_till_row, extract_from_col:extract_till_col]


# Get the super resolution image from low resolution image
def SRImage(model, lr_image, max_image_dimension=512, patch_size=96, batch_size=16):
    lr_image = lr_image[np.newaxis]

    if max(lr_image.shape) <= max_image_dimension:
        return direct_prediction(model, lr_image)

    return non_direct_prediction(model, lr_image, patch_size, batch_size)


# for large images
def non_direct_prediction(model, lr_image, patch_size, batch_size):
    patches = tf.image.extract_patches(images=lr_image,
                             sizes=[1, patch_size, patch_size, 1],
                             strides=[1, patch_size, patch_size, 1],
                             rates=[1, 1, 1, 1],
                             padding='SAME')

    patches_per_row, patches_per_col = patches.shape[1], patches.shape[2]

    patches = patches.numpy().reshape(-1, patch_size, patch_size, 3)

    results = []

    for i in range(0, patches.shape[0], batch_size):

        j = i + batch_size
        preds = model.predict(patches[i: j])
        results.append(preds)

    results = np.concatenate(results, axis=0)
    results = np.clip(results, 0, 255)
    results = np.round(results)

    joined = join_patches(patch_results, patches_per_row, patches_per_col).astype(np.uint8)

    return Image.fromarray(unpadding(lr_image, patches_per_row, patches_per_col, patch_size, joined))


def direct_prediction(model, lr_image):
    sr = model.predict(lr_image)[0]
    sr = tf.clip_by_value(sr, 0, 255)
    sr = tf.round(sr)
    sr = tf.cast(sr, tf.uint8)
    return Image.fromarray(sr.numpy())


def join_patches(patches, patches_per_row, patches_per_col):
    patch_rows, patch_cols, channels = patches[0].shape
    
    # Get the number of rows and columns
    image_rows = patch_rows * (patches_per_row - 1) + patch_rows
    image_cols = patch_cols * (patches_per_col - 1) + patch_cols
    
    joined = np.zeros((image_rows, image_cols, 3))
    
    row, col = 0, 0
    
    for patch in patches:
        joined[row * patch_rows: (row+1) * (patch_rows), col * patch_cols: (col+1) * (patch_cols)] = patch
        
        col += 1
        
        if col >= patches_per_col:
            col = 0
            row += 1
            
    return joined