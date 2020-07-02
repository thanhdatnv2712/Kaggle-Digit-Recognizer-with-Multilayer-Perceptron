import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.utils.np_utils import to_categorical
from sklearn.utils import shuffle

TRAIN_PATH = "./dataset/train.csv"
TEST_PATH = "./dataset/test.csv"
NUM_CLASSES = 10

class mnistLoader():
    def __init__(self):
        pass

    def trainLoader(self):
        csv_data = pd.read_csv(TRAIN_PATH)
        y = csv_data["label"].values
        csv_data = csv_data.drop(labels = ["label"],axis = 1)
        X = csv_data.values.reshape(-1, 28, 28, 1)
        X /= 255.0
        y = to_categorical(y, NUM_CLASSES)
        return X, y
    
    def testLoader(self):
        csv_data = pd.read_csv(TEST_PATH)
        X = csv_data.values.reshape(-1, 28, 28, 1)
        X /= 255.0
        return X

    def rawTrainLoader(self):
        csv_data = pd.read_csv(TRAIN_PATH)
        y = csv_data["label"].values
        X = csv_data.drop(labels = ["label"],axis = 1).values
        X = X.astype(np.float32)
        X /= 255.0
        y = to_categorical(y, NUM_CLASSES)
        return X, y

    def rawTestLoader(self):
        csv_data = pd.read_csv(TEST_PATH)
        X = csv_data.values
        X = X.astype(np.float32)
        X /= 255.0
        return X

    def next_batch(self, batch_size, X, y, shuffle= True):
        idx = np.arange(0, len(X))
        np.random.shuffle(idx)
        idx = idx[:batch_size]
        X_shuffle = [X[i] for i in idx]
        y_shuffle = [y[i] for i in idx]

        return np.asarray(X_shuffle), np.asarray(y_shuffle)


