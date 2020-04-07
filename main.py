# %% Hello World


import IPython.display as display
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
AUTOTUNE = tf.data.experimental.AUTOTUNE
mapping = ['三明治', '冰激淋', '土豆泥', '小米粥', '松鼠鱼', '烤冷面', '玉米饼', '甜甜圈', '芒果版戟', '鸡蛋布丁']
map_back = {k: i for i, k in enumerate(mapping)}
CLASS_NAMES = np.array(mapping)
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 128

data_dir = Path('images')

image_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255,
    validation_split=0.1
)

train_data_gen = image_generator.flow_from_directory(
    directory=str(data_dir),
    batch_size=BATCH_SIZE,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    classes=list(mapping),
    subset='training'
)

validation_data_gen = image_generator.flow_from_directory(
    directory=str(data_dir),
    batch_size=BATCH_SIZE,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    classes=list(mapping),
    subset='validation'
)

data = tf.data.Dataset.from_generator(lambda: train_data_gen, (tf.float32, tf.float32))
list_ds = tf.data.Dataset.list_files(str(data_dir / '*/*'))

# for f in list_ds.take(5):
#     print(f.numpy())

def show_batch(image_batch, label_batch):
    plt.figure(figsize=(10, 10))
    for n in range(25):
        if CLASS_NAMES[label_batch[n] == 1].size > 0:
            print('skipping!')
            continue
        print(CLASS_NAMES[label_batch[n] == 1].size)
        ax = plt.subplot(5, 5, n + 1)
        plt.imshow(image_batch[n])
        # print(CLASS_NAMES[label_batch[n] == 1])

        plt.title(CLASS_NAMES[label_batch[n] == 1][0].title())
        plt.axis('off')

image_batch, label_batch = next(train_data_gen)
show_batch(image_batch, label_batch)

def get_label(file_path):
    # convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
    print(parts[-2] == CLASS_NAMES)
    # The second to last is the class-directory
    return parts[-2] == CLASS_NAMES

def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
    return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])

def process_path(file_path):
    label = get_label(file_path)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label

# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)

# for image, label in labeled_ds.take(4):
#     print("Image shape: ", image.numpy().shape)
#     print("Label: ", label.numpy())

def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000):
    # This is a small dataset, only load it once, and keep it in memory.
    # use `.cache(filename)` to cache preprocessing work for datasets that don't
    # fit in memory.
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()
    ds = ds.shuffle(buffer_size=shuffle_buffer_size)
    # Repeat forever
    ds = ds.repeat()
    ds = ds.batch(BATCH_SIZE)
    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training.
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds

train_ds = prepare_for_training(labeled_ds)
image_batch, label_batch = next(iter(train_ds))


# sample_training_images, _ = next(data_gen)
#
# def plotImages(images_arr):
#     fig, axes = plt.subplots(1, 25, figsize=(20, 20))
#     axes = axes.flatten()
#     for img, ax in zip(images_arr, axes):
#         ax.imshow(img)
#         ax.axis('off')
#     plt.tight_layout()
#     plt.show()
#
# plotImages(sample_training_images[:25])

model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

# history = model.fit_generator(
#     train_data_gen,
#     steps_per_epoch=total_train // batch_size,
#     epochs=epochs,
#     validation_data=val_data_gen,
#     validation_steps=total_val // batch_size
# )

