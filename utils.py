import tensorflow as tf

class SaveCheckpoints(tf.keras.callbacks.Callback):
    def __init__(self, checkpoint_manager, steps_per_epoch):
        self.checkpoint_manager = checkpoint_manager
        self.steps_per_epoch = steps_per_epoch
    

    def on_epoch_end(self, epoch, logs=None):
        self.checkpoint_manager.checkpoint.epoch.assign_add(1)
        self.checkpoint_manager.checkpoint.step.assign_add(self.steps_per_epoch)
        self.checkpoint_manager.checkpoint.psnr.assign(logs["val_custom_psnr"])
        self.checkpoint_manager.save()


def custom_psnr(x1, x2):
    return tf.image.psnr(x1, x2, max_val=255)

    
def random_crop(lr_img, hr_img, hr_crop_size=96):
    '''
    this function used to take random crop from the high resolution image
    and corresponding crop of the low resolution image
    '''
    lr_crop_size = hr_crop_size // 4
    lr_shape = tf.shape(lr_img)[:2]

    lr_top = tf.random.uniform(shape=(), maxval=lr_shape[0] - lr_crop_size + 1, dtype=tf.int32)
    lr_left = tf.random.uniform(shape=(), maxval=lr_shape[1] - lr_crop_size + 1, dtype=tf.int32)

    hr_top = lr_top * 4
    hr_left = lr_left * 4

    lr_crop = lr_img[lr_top:lr_top + lr_crop_size, lr_left:lr_left + lr_crop_size]
    hr_crop = hr_img[hr_top:hr_top + hr_crop_size, hr_left:hr_left + hr_crop_size]

    return lr_crop, hr_crop