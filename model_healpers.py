import cv2
import numpy as np
import os
from random import shuffle
import pandas as pd
import csv
import tensorflow as tf
import matplotlib.pyplot as plt
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression




def get_data(basepath,img_size,encoded):
    train_images = []
    test_images = []
    csv_train = ''
    csv_test = ''
    for folder in os.listdir(basepath):
        for file in os.listdir(os.path.join(basepath, folder)):
            if (folder == "Train" and file.__contains__(".png")):
                train_images.append(os.path.join(basepath, folder, file))
            elif (folder == "Test" and file.__contains__(".png")):
                test_images.append(os.path.join(basepath, folder, file))
            elif (folder == "Test" and file.__contains__(".csv")):
                csv_test = os.path.join(basepath, folder, file)
            elif (folder == "Train" and file.__contains__(".csv")):
                csv_train = os.path.join(basepath, folder, file)

    train_pd = pd.read_csv(csv_train)
    test_pd = pd.read_csv(csv_test)

    train_dic = {}
    for ind in train_pd.index:
        train_dic[train_pd['image_name'][ind]] = train_pd['label'][ind]

    test_dic = {}
    for ind in test_pd.index:
        test_dic[test_pd['image_name'][ind]] = test_pd['label'][ind]

    train_labels = []
    for image in train_images:
        parts = os.path.split(image)

        if (train_dic.__contains__(parts[-1])):
            if (train_dic[parts[-1]] == "forged"):
                train_labels.append(encoded["forged"])
            elif (train_dic[parts[-1]] == "real"):
                train_labels.append(encoded["real"])
        else:
            train_labels.append(encoded["forged"])

    test_labels = []
    for image in test_images:
        parts = os.path.split(image)

        if (test_dic.__contains__(parts[-1])):
            if (test_dic[parts[-1]] == "forged"):
                test_labels.append(encoded["forged"])
            elif (test_dic[parts[-1]] == "real"):
                test_labels.append(encoded["real"])
        else:
            test_labels.append(encoded["forged"])

    training_data = []
    for image in train_images:
        training_data.append(cv2.resize(cv2.imread(image, 0), (img_size, img_size)))

    test_data = []
    for image in test_images:
        test_data.append(cv2.resize(cv2.imread(image, 0), (img_size, img_size)))

    # cv2.imshow("train 1",training_data[0])
    # cv2.waitKey(5000)

    training_data = np.array(training_data).reshape(-1, img_size, img_size, 1)
    train_labels = np.array(train_labels).reshape(-1, 2)
    test_data = np.array(test_data).reshape(-1, img_size, img_size, 1)
    test_labels = np.array(test_labels).reshape(-1, 2)

    return training_data,train_labels,test_data,test_labels


def create_model(img_size):
    conv_input = input_data(shape=[None, img_size, img_size, 1], name='input')
    conv1 = conv_2d(conv_input, 128, 3, strides=1, padding='same', activation='relu')
    pool1 = max_pool_2d(conv1, 3)

    conv2 = conv_2d(pool1, 64, 3, strides=1, padding='same', activation='relu')
    pool2 = max_pool_2d(conv2, 3)

    # conv3 = conv_2d(pool2, 32, 6,strides=1,padding='same',activation='relu')
    # pool3 = max_pool_2d(conv3, 3)

    fully_layer1 = fully_connected(pool2, 64, activation='relu')
    fully_layer2 = fully_connected(fully_layer1, 32, activation='relu')

    cnn_layers = fully_connected(fully_layer2, 2, activation='softmax')

    cnn_layers = regression(cnn_layers, optimizer='adam', learning_rate=0.0001, loss='categorical_crossentropy',
                            name='targets')
    model = tflearn.DNN(cnn_layers, tensorboard_dir='log', tensorboard_verbose=3)

    return model



#
# def mean_absolute_percentage_error(y_test, estimation_result):
#     y_test, estimation_result = np.array(y_test), np.array(estimation_result)
#     return np.mean(np.abs((y_test - estimation_result) / y_test)) * 100

def test_script(img,model):
    predict=model.predict_label(img)
    return predict