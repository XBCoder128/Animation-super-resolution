import tensorflow as tf
from tensorflow import keras
import config as conf


def l1_loss(y_true, y_pred):
    return keras.backend.mean(keras.backend.abs(y_true - y_pred))


def get_model(input_shape=None):  # , output_shape=None
    if input_shape is None:
        input_shape = [conf.train_input_size, conf.train_input_size, 3]
    # if output_shape is None:
    #     output_shape = [conf.target_size, conf.target_size, 3]

    model_input = keras.Input(input_shape)  # 128 * 128 * 3   class

    level_1_conv1 = keras.layers.Conv2D(64, 3, 1, 'same', activation='relu')(model_input)
    level_1_conv2 = keras.layers.Conv2D(64, 3, 1, 'same', activation='relu')(level_1_conv1)
    level_1_pool = keras.layers.MaxPooling2D()(level_1_conv2)
    # 64 * 64 * 64

    level_2_conv1 = keras.layers.Conv2D(128, 3, 1, 'same', activation='relu')(level_1_pool)
    level_2_conv2 = keras.layers.Conv2D(128, 3, 1, 'same', activation='relu')(level_2_conv1)
    level_2_pool = keras.layers.MaxPooling2D()(level_2_conv2)
    # 32 * 32 * 128

    level_3_conv1 = keras.layers.Conv2D(256, 3, 1, 'same', activation='relu')(level_2_pool)
    level_3_conv2 = keras.layers.Conv2D(256, 3, 1, 'same', activation='relu')(level_3_conv1)
    level_3_pool = keras.layers.MaxPooling2D()(level_3_conv2)
    # 16 * 16 * 256

    level_4_conv1 = keras.layers.Conv2D(512, 3, 1, 'same', activation='relu')(level_3_pool)
    level_4_conv2 = keras.layers.Conv2D(512, 3, 1, 'same', activation='relu')(level_4_conv1)
    level_4_pool = keras.layers.MaxPooling2D()(level_4_conv2)
    # 8 * 8 * 512

    level_3_conv1t = keras.layers.Conv2DTranspose(256, 3, 2, 'same', activation='elu')(level_4_pool)
    level_3_concat = keras.layers.Concatenate()([level_3_conv1t, level_4_conv2])
    level_3_conv2t = keras.layers.Conv2DTranspose(256, 3, 1, 'same', activation='elu')(level_3_concat)

    level_2_conv1t = keras.layers.Conv2DTranspose(128, 3, 2, 'same', activation='elu')(level_3_conv2t)
    level_2_concat = keras.layers.Concatenate()([level_2_conv1t, level_3_conv2])
    level_2_conv2t = keras.layers.Conv2DTranspose(128, 3, 1, 'same', activation='elu')(level_2_concat)

    level_1_conv1t = keras.layers.Conv2DTranspose(64, 3, 2, 'same', activation='elu')(level_2_conv2t)
    level_1_concat = keras.layers.Concatenate()([level_1_conv1t, level_2_conv1])
    level_1_conv2t = keras.layers.Conv2DTranspose(64, 3, 1, 'same', activation='elu')(level_1_concat)

    level_0_conv1t = keras.layers.Conv2DTranspose(64, 3, 2, 'same', activation='elu')(level_1_conv2t)
    level_0_concat = keras.layers.Concatenate()([level_0_conv1t, level_1_conv1])
    level_0_conv2t = keras.layers.Conv2DTranspose(64, 3, 2, 'same', activation='elu')(level_0_concat)
    # 256 * 256 * 64

    level_0_conv3t = keras.layers.Conv2D(64, 3, 1, 'same', activation='elu')(level_0_conv2t)
    level_0_conv4t = keras.layers.Conv2D(64, 3, 1, 'same', activation='linear')(level_0_conv3t)
    model_output = keras.layers.Conv2D(3, 3, 1, 'same', activation='elu')(level_0_conv4t)

    model = keras.Model(inputs=model_input, outputs=model_output)
    model.compile(loss=l1_loss, optimizer=keras.optimizers.RMSprop(learning_rate=0.005))
    model.summary()
    return model
