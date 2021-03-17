import tensorflow as tf
import matplotlib.pyplot as plt
import os


@tf.function
def trans_func(file_path):
    img_bin = tf.io.read_file(file_path)
    img_decode = tf.cond(
        tf.image.is_jpeg(img_bin),
        lambda: tf.image.decode_jpeg(img_bin, channels=3),
        lambda: tf.image.decode_png(img_bin, channels=3)
    )

    tf.image.random_crop()

    img = tf.image.convert_image_dtype(img_decode, tf.float32)
    img = tf.image.resize(img, (256, 256))
    return img


# 像素


file_list = [os.path.join('dataset', file) for file in os.listdir('dataset')]
ds = tf.data.Dataset.from_tensor_slices(file_list)
ds = ds.repeat(1)
ds = ds.shuffle(64)
ds = ds.map(trans_func, num_parallel_calls=-1, deterministic=False)
ds = ds.batch(8)

for it in ds.take(5):
    print(it.shape)
