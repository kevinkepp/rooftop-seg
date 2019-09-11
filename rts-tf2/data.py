import math

import tensorflow as tf
import tensorflow_addons as tfa

AUTOTUNE = tf.data.experimental.AUTOTUNE


def get_data_paths(data_dir, mode):
    return list(str(p) for p in (data_dir / "images" / mode).glob("*.png")), \
           list(str(p) for p in (data_dir / "labels" / mode).glob("*.png"))


def get_dataset(data_dir, mode, batch_size):
    image_paths, mask_paths = get_data_paths(data_dir, mode)
    nb_samples = len(image_paths)
    images = tf.data.Dataset.from_tensor_slices(image_paths).map(load_image)
    masks = tf.data.Dataset.from_tensor_slices(mask_paths).map(load_mask)
    ds = tf.data.Dataset.zip((images, masks))
    if mode == "train":
        ds = ds.repeat() \
            .map(augment_train, num_parallel_calls=AUTOTUNE) \
            .shuffle(buffer_size=batch_size * 4) \
            .batch(batch_size) \
            .prefetch(buffer_size=AUTOTUNE)
    else:  # mode == "val"
        ds = ds.map(preprocess_val).batch(nb_samples)
    return ds, nb_samples


def get_test_dataset(data_dir):
    input_image_paths, _ = get_data_paths(data_dir, "test")
    nb_samples = len(input_image_paths)
    input_images = tf.data.Dataset.from_tensor_slices(input_image_paths) \
        .map(load_image) \
        .map(preprocess_test) \
        .batch(nb_samples)
    return input_images, nb_samples


def load_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_png(image, channels=3)
    image /= 255
    return image


def load_mask(path):
    mask = tf.io.read_file(path)
    mask = tf.image.decode_png(mask, channels=1)
    mask /= 255
    return mask


@tf.function
def augment_train(image, mask):
    input_size = tf.convert_to_tensor((256, 256))
    target_size = tf.convert_to_tensor((224, 224))

    # zoom in our out
    if tf.random.uniform(()) > 0.5:
        scale = tf.random.uniform((), minval=0.6, maxval=1.5)
        input_size = tf.cast(tf.cast(input_size, tf.float32) * scale, tf.int32)
        image = tf.image.resize(image, size=input_size)
        mask = tf.image.resize(mask, size=input_size)

    # crop or pad
    if input_size[0] > target_size[0]:
        image, mask = random_crop(image, mask, target_size)
    elif input_size[0] < target_size[1]:
        pad_h = target_size[0] - input_size[0]
        pad_w = target_size[1] - input_size[1]
        offset_h = tf.cast(tf.random.uniform((), maxval=tf.cast(pad_h, tf.float32)), tf.int32)
        offset_w = tf.cast(tf.random.uniform((), maxval=tf.cast(pad_w, tf.float32)), tf.int32)
        image = tf.image.pad_to_bounding_box(image, offset_h, offset_w, target_size[0], target_size[1])
        mask = tf.image.pad_to_bounding_box(mask, offset_h, offset_w, target_size[0], target_size[1])

    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_up_down(image)
        mask = tf.image.flip_up_down(mask)
    if tf.random.uniform(()) > 0.5:
        angle = tf.random.uniform((), minval=-math.pi, maxval=math.pi)
        image = tfa.image.rotate(image, angle)
        mask = tfa.image.rotate(mask, angle)

    return image, mask


def get_random_2d_offset(image, target_size):
    src_size = tf.shape(image)[:2]
    limit = src_size - target_size + 1
    return tf.random.uniform(tf.shape(src_size), dtype=tf.int32, maxval=tf.int32.max) % limit


def crop_2d_offset(image, offset_2d, target_size):
    channels = tf.expand_dims(tf.shape(image)[2], axis=0)
    img_offset = tf.concat((offset_2d, tf.constant(0, shape=(1,))), axis=0)
    target_shape = tf.concat((target_size, channels), axis=0)
    return tf.slice(image, img_offset, target_shape)


def random_crop(image, mask, target_size):
    # compute (row, col) offset
    offset_2d = get_random_2d_offset(image, target_size)
    # crop offset without changing channel dim
    image_crop = crop_2d_offset(image, offset_2d, target_size)
    mask_crop = crop_2d_offset(mask, offset_2d, target_size)
    return image_crop, mask_crop


def preprocess_val(image, mask):
    image = tf.image.resize(image, (224, 224))
    mask = tf.image.resize(mask, (224, 224))
    return image, mask


def preprocess_test(input_image):
    return tf.image.resize(input_image, (224, 224))


# keras provides nice image pre-processing with ImageDataGenerator, but
# while generating equally augmented (image, label) pairs using the same random seed,
# this break when using a second ImageDataGenerator for the  validation set
def get_dataset_gen(data_dir, mode, batch_size, seed=1):
    if mode == "train":
        data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode="constant",
            cval=0)
    else:
        data_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
    images = data_gen.flow_from_directory(
        str(data_dir / "images" / mode),
        class_mode=None,
        color_mode="rgb",
        target_size=(224, 224),
        batch_size=batch_size,
        shuffle=True,
        seed=seed)
    masks = data_gen.flow_from_directory(
        str(data_dir / "labels" / mode),
        class_mode=None,
        color_mode="grayscale",
        target_size=(224, 224),
        batch_size=batch_size,
        shuffle=True,
        seed=seed)
    # create generator as workaround because TF2.0 does not recognize zip object as generator
    return (pair for pair in zip(images, masks))


def get_test_dataset_gen(data_dir):
    data_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
    images = data_gen.flow_from_directory(
        str(data_dir / "images" / "test"),
        class_mode=None,
        color_mode="rgb",
        target_size=(224, 224),
        batch_size=5)
    return images
