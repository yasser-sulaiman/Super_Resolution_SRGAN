import os
import tensorflow as tf
import glob
from tensorflow.python.data.experimental import AUTOTUNE

# Create a Dataset Fromdirectory
def images_from_directory(data_directory, image_directory):

    images_path = os.path.join(data_directory, image_directory)
    file_names = sorted(glob.glob(images_path + "/*.png"))

    dataset = tf.data.Dataset.from_tensor_slices(file_names)
    dataset = dataset.map(tf.io.read_file)
    dataset = dataset.map(lambda x: tf.image.decode_png(x, channels=3), num_parallel_calls=AUTOTUNE)

    cache_directory = os.path.join(data_directory, "cache", image_directory)

    os.makedirs(cache_directory, exist_ok=True)

    cache_file = cache_directory + "/cache"

    dataset = dataset.cache(cache_file)

    if not os.path.exists(cache_file + ".index"):
        populate_cache(dataset, cache_file)

    return dataset

# make a training dataset
def training_dataset(dataset_parameters, train_mappings, batch_size):
    low_res_dataset = images_from_directory(dataset_parameters['data_directory'], dataset_parameters['train_low_res'])
    hi_res_dataset = images_from_directory(dataset_parameters['data_directory'], "HR/DIV2K_train_HR")

    dataset = tf.data.Dataset.zip((low_res_dataset, hi_res_dataset))

    for mapping in train_mappings:
        dataset = dataset.map(mapping, num_parallel_calls=AUTOTUNE)

    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    return dataset

# make a validation dataset
def validation_dataset(dataset_parameters):
    lr_dataset = images_from_directory(dataset_parameters['data_directory'], dataset_parameters['val_low_res'])
    hr_dataset = images_from_directory(dataset_parameters['data_directory'], "HR/DIV2K_valid_HR")

    dataset = tf.data.Dataset.zip((lr_dataset, hr_dataset))

    dataset = dataset.batch(1)
    dataset = dataset.repeat(1)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    return dataset

# Create traaining and validation datasets togather
def create_training_and_validation_datasets(dataset_parameters, train_mappings, train_batch_size=16):
    training_dataset_ = training_dataset(dataset_parameters, train_mappings, train_batch_size)
    validation_dataset_ = validation_dataset(dataset_parameters)

    return training_dataset_, validation_dataset_

# papulate cache if not exist for faster load
def populate_cache(dataset, cache_file):
    print(f'Begin caching in {cache_file}.')
    for _ in dataset: pass
    print(f'Completed caching in {cache_file}.')