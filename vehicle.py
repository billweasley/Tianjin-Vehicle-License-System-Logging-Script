# -*- coding: utf-8 -*-
phone = ''
password = ''
ftqq_id = ''

import os

model_path = os.path.dirname(os.path.abspath(__file__)) + "/"
print(model_path)

import matplotlib.image as mpimg
import requests
from bs4 import BeautifulSoup
import random
import shutil
import numpy as np
import string
import datetime
import tensorflow.compat.v1 as tf
import tensorflow.compat.v1.keras.backend as K
from tensorflow.compat.v1.keras.models import *
from tensorflow.compat.v1.keras.layers import *
import logging
tf.get_logger().setLevel(logging.ERROR)
tf.disable_v2_behavior()
print("Script starting...")
characters = string.digits + string.ascii_uppercase
width, height, n_len, n_class = 70, 23, 4, len(characters) + 1


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    y_pred = tf.Print(y_pred, [y_pred], "y_pred")
    labels = tf.Print(labels, [labels], "labels")
    input_length = tf.Print(input_length, [input_length], "input_length")
    label_length = tf.Print(label_length, [label_length], "label_length")
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


input_tensor = Input((height, width, 3))
x = input_tensor
for i, n_cnn in enumerate([2, 2, 2, 2]):
    for j in range(n_cnn):
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

base_model = Model(inputs=input_tensor, outputs=x)

labels = Input(name='the_labels', shape=[n_len], dtype='float32')
input_length = Input(name='input_length', shape=[1], dtype='int64')
label_length = Input(name='label_length', shape=[1], dtype='int64')
print_node = tf.Print(x, [x], "shape of output")
loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([print_node, labels, input_length, label_length])

model = Model(inputs=[input_tensor, labels, input_length, label_length], outputs=loss_out)
model.load_weights(model_path + 'ctc_best.h5')

table = None
out = None

while table is None:
    s = requests.Session()
    r = s.get("http://apply.xkctk.jtys.tj.gov.cn/apply/validCodeImage.html?r=" + str(random.random()) + "&ee=1",
              stream=True)
    with open(model_path + '0000000.jpg', 'wb') as out_file:
        shutil.copyfileobj(r.raw, out_file)
    img = np.expand_dims(mpimg.imread(model_path + '0000000.jpg') / 255, axis=0)
    characters2 = characters + ' '
    y_pred = base_model.predict(img)
    out = K.get_value(K.ctc_decode(y_pred, input_length=np.ones(y_pred.shape[0]) * y_pred.shape[1], )[0][0])[:, :4]
    out = ''.join([characters[x] for x in out[0]])
    url = 'http://apply.xkctk.jtys.tj.gov.cn/apply/user/person/login.html?r=' + str(random.random())
    values = {'loginType': 'MOBILE',
              'type': 'person',
              'logInFrom': '0',
              'grSelect': 'MOBILE',
              'mobile': phone,
              'password': password,
              'validCode': out,
              'unitLoginTypeSelect': 'MOBILE',
              'sySelect': 'MOBILE'
              }
    headers = {
        'Host': 'apply.xkctk.jtys.tj.gov.cn',
        'Connection': 'keep-alive',
        'Content-Length': '156',
        'Pragma': 'no-cache',
        'Cache-Control': 'no-cache',
        'Upgrade-Insecure-Requests': '1',
        'Origin': 'http://xkctk.jtys.tj.gov.cn',
        'Content-Type': 'application/x-www-form-urlencoded',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/81.0.4044.138 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,'
                  'application/signed-exchange;v=b3;q=0.9',
        'Referer': 'http://xkctk.jtys.tj.gov.cn/?',
        'Accept-Encoding': 'gzip, deflate',
        'Accept-Language': 'zh-CN,zh;q=0.9,zh-TW;q=0.8,en;q=0.7',
        'Cookie': 'JSESSIONID=' + s.cookies.get_dict()['JSESSIONID']
    }

    r2 = s.post(url, data=values, headers=headers)

    soup = BeautifulSoup(r2.text, features="html.parser")
    table = soup.find('table', {"class": "sub_table1"})
    if table is None:
        print("Table is none. This is probably because the verification code cannot be recognised by model.")

table_body = table.find('tbody')
data = []
rows = table_body.find_all('tr')
hds = table_body.find_all('th')
header_cols = [ele.text.strip() for ele in hds[:-1]]
for row in rows:
    cols = row.find_all('td')
    cols = [ele.text.strip() for ele in cols]
    if cols:
        data.append([ele for ele in cols[:-1] if ele])  # Get rid of empty values

headers_form = {
    'Host': 'apply.xkctk.jtys.tj.gov.cn',
    'Connection': 'keep-alive',
    'Pragma': 'no-cache',
    'Cache-Control': 'no-cache',
    'Upgrade-Insecure-Requests': '1',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/81.0.4044.138 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,'
              'application/signed-exchange;v=b3;q=0.9',
    'Referer': 'http://apply.xkctk.jtys.tj.gov.cn/apply/user/person/manage.do?',
    'Accept-Encoding': 'gzip, deflate',
    'Accept-Language': 'zh-CN,zh;q=0.9,zh-TW;q=0.8,en;q=0.7',
    'Cookie': 'JSESSIONID=' + s.cookies.get_dict()['JSESSIONID']
}

r3 = s.get("http://apply.xkctk.jtys.tj.gov.cn" + table_body.select('a')[0].get('href'), headers=headers_form)

soup2 = BeautifulSoup(r3.text, features="html.parser")
dl = soup2.find('dl', {"class": "state"})
application_state = dl.find_all('dt')[0].text.replace("\r", "").replace("\t", "").replace("\n", "").strip().split("： ")
header_cols.append(application_state[0])
data[0].append(application_state[1])
info_dict = dict(zip(header_cols, data[0]))
notification_info = str({
        "请求时间": str(datetime.datetime.now()),
        "申请时间": info_dict["申请时间"],
        "申请状态": info_dict["申请状态"]
 })
print(notification_info)
requests.get("https://sc.ftqq.com/{0}.send?text={1}&desp={2}".format(ftqq_id, "小汽车摇号自动登录", notification_info))