import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

import tensorflow as tf

import pandas as pd
import numpy as np
from mnistLoader import mnistLoader
from sklearn.model_selection import train_test_split

tf.reset_default_graph()
keras.backend.clear_session()

loader = mnistLoader()
X, y = loader.rawTrainLoader()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
X_test_ = loader.rawTestLoader()

def save_csv(y):
    ImageId = list(np.asarray(list(range(28000))) + 1)
    Label = list(y)
    dict_value = {
        "ImageId" : ImageId,
        "Label" : Label
    }
    pf = pd.DataFrame(dict_value, columns=["ImageId", "Label"])
    pf.to_csv(" keras-mnist.csv", index= None)

num_layers = 2
num_neurons = []
num_inputs = 784
num_outputs = 10

for i in range(num_layers):
    num_neurons.append(256)

learning_rate = 0.01
n_epochs = 50
batch_size= 32

model= Sequential()
model.add(Dense(units=num_neurons[0], activation="relu",
            input_shape=(num_inputs,)))
model.add(Dense(units=num_neurons[1], activation="relu"))
model.add(Dense(units=num_outputs, activation="softmax"))
model.summary()
model.compile(loss="categorical_crossentropy",
                optimizer= SGD(learning_rate),
                metrics=['accuracy'])

# model.fit(X_train, y_train, batch_size, epochs=n_epochs)
# model.save("weights/keras-mnist.hdf5")
# score = model.evaluate(X_test, y_test)
# print('\nTest loss:', score[0])
# print('Test accuracy:', score[1])

model.load_weights("weights/keras-mnist.hdf5")
ans = model.predict(X_test_)
ans = np.argmax(ans, axis=1)
save_csv(ans)



