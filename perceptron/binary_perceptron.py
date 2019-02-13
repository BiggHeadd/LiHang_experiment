# -*- coding:utf-8 -*-
# Edited by bighead 19-1-29

import numpy as np
import pandas as pd
import time
import random
from utils import get_train_data, get_hog_feature
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

########## params
feature_length = 324
object_num = 0
learning_rate = 0.0001
study_total = 10000
########## END params

def train(inputs, labels):
    """ Training a perception model of inputs and it's labels, and return
    the weights and biases of the model

    Args:
         inputs: numpy array of the input features. shape->('inputs :', (42000, 324))
         labels: numpy array of the input labels. shape->('labels :', 42000)

    return:
         w: the weights of perception model
         b: the biases of perception model
    """
    ########## Training
    trainset_length = len(labels)
    ##### init w, b
    w = np.zeros((feature_length, 1))
    b = 0
    ##### END init w, b

    ##### Some flag
    study_count = 0          # count when the perception predict wrong
    nochange_count = 0       # count the ringht prediction combo
    nochange_upper_limit = 100000 # the upper limit of the predicton combo
    ##### END Some flag

    while True:
        if nochange_count >= nochange_upper_limit:
            break

        index = random.randint(0, trainset_length-1)
        img = inputs[index]
        label = labels[index]

        yi = int(label != object_num)*2-1
        result = yi * (np.dot(img, w) + b)
        # print(label, object_num, yi)

        img = np.reshape(inputs[index], (feature_length, 1))
        if result <= 0:
            w += img * yi * learning_rate
            b += yi * learning_rate

            study_count += 1
            if study_count >= study_total:
                break

            nochange_count=0

        nochange_count += 1
    ########## END Training
    return w, b

def predict(test_set, w, b):
    """use the trained perception model weights and biases to predict the test set
    and return the predict labels

    Args:
         test_set: numpy array of test inputs. shape(324, ?)
         w: weights
         b: biases

    return:
         predictions: numpy array of test's predict
    """
    predictions = []
    for img in test_set:
        result = np.dot(img, w) + b
        result = result > 0

        predictions.append(result)

    predictions = np.array(predictions)
    return predictions


########## Get inputs and labels
read_start = time.time()
labels, inputs = get_train_data("../data/train_binary.csv")
inputs = get_hog_feature(inputs)
####### NOTE:
### ('labels :', 42000)
### ('inputs :', (42000, 324))
####### END NOTE('labels :', 42000)
########## END Get inputs and labels

########## Split into train and test set
train_inputs, test_inputs, train_labels, test_labels = train_test_split(inputs, labels, test_size=0.3, random_state=2333)
read_end = time.time()
########## END Split into train and test set

########## read_data log
print("read data finished!!")
print("cost time: {}".format(read_end-read_start))
########## END read_data log

########## TRAINING
print("\ntraining........")
train_start = time.time()
w,b = train(train_inputs, train_labels)
train_end = time.time()
print("\nfinish")
print("cost time: {}".format(train_end-train_start))
########## END TRAINING

########## PREDICTING
print("\npredicting........")
predict_start = time.time()
predictions = predict(test_inputs, w, b)
predict_end = time.time()
print("\nfinish")
print("cost time: {}".format(predict_end-predict_start))
########## END PREDICTING

########## ACCURACY
score = accuracy_score(test_labels, predictions)
print("accuracy_score: {}".format(score))
########## END ACCURACY
