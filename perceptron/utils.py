# -*- coding:utf-8 -*-
# Edited by bighead 19-1-28

import pandas as pd

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
    print("labels", len(labels))
    return (labels, pixels)


if __name__ == "__main__":
    get_train_data("data/train_binary.csv")
