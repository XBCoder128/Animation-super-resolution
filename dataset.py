import tensorflow as tf
import config as conf
import os


@tf.function
def trans_func(files):
    img_bin = tf.io.read_file(files)
    img_decode = tf.cond(
        tf.image.is_jpeg(img_bin),
        lambda: tf.image.decode_jpeg(img_bin, channels=3),
        lambda: tf.image.decode_png(img_bin, channels=3)
    )

    img_size = tf.cast(tf.shape(img_decode), dtype=tf.float32)
    # tf.cast 强制类型转换

    min_size = tf.math.minimum(x=img_size[0], y=img_size[1])  # 取出较小的一边
    crop_size = tf.random.uniform([], min_size * conf.resize_min_scale, min_size * conf.resize_max_scale)

    # label
    target_img = tf.image.convert_image_dtype(img_decode, tf.float32)  # 归一化
    target_img = tf.image.random_crop(target_img, [256, 256, 3])  # -》 带有shape的tensor

    target_resize_mhd = 'area'
    train_resize_mhd = 'nearest'

    target_img = tf.image.resize(target_img, [conf.target_size, conf.target_size], method=target_resize_mhd)  # 缩放到固定大小
    target_img = tf.image.random_flip_left_right(target_img)  # 随机左右翻转
    target_img = tf.image.random_contrast(target_img, 0.25, 0.75)
    target_img = tf.image.random_brightness(target_img, 0.2)
    # target_img = tf.image.random_hue(target_img, 0.1)
    # target_img = tf.image.random_saturation(target_img,0.3, 0.7)

    train_img = tf.image.resize(target_img, [conf.train_input_size, conf.train_input_size],
                                method=train_resize_mhd)  # 缩放到固定大小
    train_img = tf.image.random_jpeg_quality(train_img, min_jpeg_quality=conf.jpge_quality_min,
                                             max_jpeg_quality=conf.jpge_quality_max)
    noise = tf.random.normal(shape=[conf.train_input_size, conf.train_input_size, 3], stddev=0.015)
    train_img = train_img + noise

    return [train_img, target_img]


def get_dataset():
    files_list = [os.path.join(conf.dataset_path, name) for name in os.listdir(conf.dataset_path)]
    ds_ = tf.data.Dataset.from_tensor_slices(files_list)
    ds_ = ds_.repeat(conf.train_repeat)
    ds_ = ds_.shuffle(128)

    # repeat一定要在map之前
    ds_ = ds_.map(trans_func, num_parallel_calls=-1, deterministic=False)
    ds_ = ds_.batch(conf.batch_size)
    return ds_


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    ds = get_dataset()
    for (train, target) in ds:
        print(train.shape, target.shape)
        plt.figure(figsize=(12, 6), dpi=80)
        plt.subplot(121)
        plt.imshow(train[0])
        plt.subplot(122)
        plt.imshow(target[0])
        plt.show()


