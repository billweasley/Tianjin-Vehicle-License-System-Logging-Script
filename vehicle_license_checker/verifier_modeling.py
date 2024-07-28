import os
import string

import matplotlib.image as mpimg
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow.compat.v1.keras.backend as K
from tensorflow.compat.v1.keras.models import Model
from tensorflow.compat.v1.keras.layers import (
    Input,
    Conv2D, 
    BatchNormalization, 
    Activation,
    MaxPooling2D,
    TimeDistributed,
    Permute,
    Flatten,
    LSTMCell,
    Bidirectional,
    Dense,
    RNN,
    Lambda
)
import logging
#import cv2

tf.disable_v2_behavior()
tf.get_logger().setLevel(logging.ERROR)

class Verifier:
    def __init__(self, model_path: str = None) -> None:
        self.model_path = model_path
        self.model, self.base_model = self._init_ocr_model(model_path=model_path)
        self.characters = string.digits + string.ascii_uppercase

    def ctc_lambda_func(self, args):
        y_pred, labels, input_length, label_length = args
        y_pred = tf.Print(y_pred, [y_pred], "y_pred")
        labels = tf.Print(labels, [labels], "labels")
        input_length = tf.Print(input_length, [input_length], "input_length")
        label_length = tf.Print(label_length, [label_length], "label_length")
        return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

    def _init_ocr_model(self, model_path: str = None):

        characters = string.digits + string.ascii_uppercase
        width, height, n_len, n_class = 70, 23, 4, len(characters) + 1
        input_tensor = Input((height, width, 3))
        x = input_tensor
        for i, n_cnn in enumerate([2, 2, 2, 2]):
            for _ in range(n_cnn):
                x = Conv2D(32 * 2 ** min(i, 3), kernel_size=3, padding='same', kernel_initializer='he_uniform')(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)
            x = MaxPooling2D(2 if i < 2 else (2, 1))(x)
        x = Permute((2, 1, 3))(x)
        x = TimeDistributed(Flatten())(x)
        rnn_size = 128
        x = Bidirectional(RNN(LSTMCell(rnn_size, recurrent_activation='sigmoid'), return_sequences=True))(x)
        x = Bidirectional(RNN(LSTMCell(rnn_size, recurrent_activation='sigmoid'), return_sequences=True))(x)
        x = Dense(n_class, activation='softmax')(x)
        print_node =  tf.Print(x, [x], "shape of output")
        labels = Input(name='the_labels', shape=[n_len], dtype='float32')
        input_length = Input(name='input_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')
        loss_out = Lambda(self.ctc_lambda_func, output_shape=(1,), name='ctc')([print_node, labels, input_length, label_length])
        base_model = Model(inputs=input_tensor, outputs=x)
        model = Model(inputs=[input_tensor, labels, input_length, label_length], outputs=loss_out)

        if model_path is not None:
            model.load_weights(model_path)

        return model, base_model

    def get_result(self, img_path: str) -> str:
    #    img_path = self.__convert_to_jpg(img_path)
        img = np.expand_dims(mpimg.imread(img_path) / 255, axis=0)
        print("img shape", img.shape)
        y_pred = self.base_model.predict(img)
        print("y_pred", y_pred)
        out = K.get_value(K.ctc_decode(y_pred, input_length=np.ones(y_pred.shape[0]) * y_pred.shape[1], )[0][0])[:, :4]
        print(f"out {out}")
        out = ''.join([self.characters[x] for x in out[0]])

    #def __convert_to_jpg(self, img_path: str):
    #    image = cv2.imread(img_path)
    #    write_path = os.path.splitext(img_path)[0]+'.jpg'
    #    cv2.imwrite(write_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    #    return write_path