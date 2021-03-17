import tensorflow as tf
import config as conf
import dataset
import model
import matplotlib.pyplot as plt
import os

ds = dataset.get_dataset()
mymodel = model.get_model([conf.train_input_size, conf.train_input_size, 3])
if os.path.exists('model.h5'):
    print('加载权重...')
    mymodel.load_weights('model.h5')

# try:
#     mymodel.fit(ds)
# except KeyboardInterrupt:
#     pass
#     # ctrl + C
# finally:
#     print('保存权重...')
#     mymodel.save_weights('model.h5')

ds = dataset.get_dataset()
for a, b in ds:
    print(a, b)
    a = a[0:1]
    b = b[0:1]
    output = mymodel(a)
    plt.figure(figsize=(18, 6), dpi=80)

    plt.subplot(131)
    plt.imshow(a[0])
    plt.title('input')

    plt.subplot(132)
    plt.imshow(output[0])
    plt.title('predict')

    plt.subplot(133)
    plt.imshow(b[0])
    plt.title('target')

    plt.show()

# test_pic_bin = tf.io.read_file('./1.jpg')
# test_pic = tf.io.decode_image(test_pic_bin, channels=3)
# test_pic = tf.image.convert_image_dtype(test_pic, tf.float32)
# test_pic = tf.expand_dims(test_pic, axis=0)
#
# output = mymodel(test_pic)
# plt.imshow(output[0])
# plt.show()
#
# output_image = tf.image.convert_image_dtype(output, dtype=tf.uint8)
#
# output_image = tf.image.encode_png(output_image[0])
# tf.io.write_file('test.png', output_image)
