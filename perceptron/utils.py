# -*- coding:utf-8 -*-
# Edited by bighead 19-1-28

import pandas as pd
import numpy as np
import cv2


def get_train_data(data_csv):
    """the data is store in the data_csv files, using pandas get the labels and pixelsself.
       and return a numpy like labels and pixel data

       Args:


            data_csv: the path of the csv file containing the training data

       Return:
            (labels, pixels)
            labels and pixels which is numpy array
            labels: [1 0 1 ... 1 1 1]
            pixels: [[0 0 0 ... 0 0 0]
                    [0 0 0 ... 0 0 0]
                    [0 0 0 ... 0 0 0]
                    ...
                    [0 0 0 ... 0 0 0]
                    [0 0 0 ... 0 0 0]
                    [0 0 0 ... 0 0 0]]
    """
    binary_data = pd.read_csv(data_csv, header=0)
    binary_data = binary_data.values
    labels = binary_data[:, 0]
    pixels = binary_data[:, 1:]
    return (labels, pixels)


def get_hog_feature(pixels):
    """get the inputs sets of the numbers pictures, get hog features from each pic and return
    a new set of features

    Args:
         pixels: the numpy array like data of inputs

    returns:
         features: the hog features of the origin pic features
            the shape change from (28, 28) to (18, 18)
    """
    features = []
    hog = cv2.HOGDescriptor('data/hog.xml')
    for img in pixels:
        img = np.reshape(img, (28, 28))
        cv_img = img.astype(np.uint8)

        hog_feature = hog.compute(cv_img)
        features.append(hog_feature)

    features = np.array(features)
    features = np.reshape(features, (-1, 324))

    return features


if __name__ == "__main__":
    labels, inputs = get_train_data("data/train_binary.csv")
    get_hog_feature(inputs)
