""" LSTM Network.
A RNN Network (LSTM) implementation example using Keras.
This example is using the MNIST handwritten digits dataset (http://yann.lecun.com/exdb/mnist/)

Ressouces:
    [Long Short Term Memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf)
    [Understanding LSTMs](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
    [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)

Repository: https://github.com/ar-ms/lstm-mnist
"""

# Imports
import sys

from tensorflow.examples.tutorials.mnist import input_data
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.models import load_model
import numpy as np


class MnistLSTMClassifier(object):
    def __init__(self):
        # Classifier
        self.time_steps=28 # timesteps to unroll
        self.n_units=128 # hidden LSTM units
        self.n_inputs=28 # rows of 28 pixels (an mnist img is 28x28)
        self.n_classes=10 # mnist classes/labels (0-9)
        self.batch_size=128 # Size of each batch
        self.n_epochs=5
        # Internal
        self._data_loaded = False
        self._trained = False

    def __create_model(self):
        self.model = Sequential()
        self.model.add(LSTM(self.n_units, input_shape=(self.time_steps, self.n_inputs)))
        self.model.add(Dense(self.n_classes, activation='softmax'))

        self.model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])

    def __load_data(self):
        self.mnist = input_data.read_data_sets("mnist", one_hot=True)
        self._data_loaded = True

    def train(self, save_model=False):
        self.__create_model()
        if self._data_loaded == False:
            self.__load_data()

        x_train = [x.reshape((-1, self.time_steps, self.n_inputs)) for x in self.mnist.train.images]
        x_train = np.array(x_train).reshape((-1, self.time_steps, self.n_inputs))

        self.model.fit(x_train, self.mnist.train.labels,
                  batch_size=self.batch_size, epochs=self.n_epochs, shuffle=False)

        self._trained = True
        
        if save_model:
            self.model.save("./saved_model/lstm-model.h5")

    def evaluate(self, model=None):
        if self._trained == False and model == None:
            errmsg = "[!] Error: classifier wasn't trained or classifier path is not precised."
            print(errmsg, file=sys.stderr)
            sys.exit(0)

        if self._data_loaded == False:
            self.__load_data()

        x_test = [x.reshape((-1, self.time_steps, self.n_inputs)) for x in self.mnist.test.images]
        x_test = np.array(x_test).reshape((-1, self.time_steps, self.n_inputs))

        model = load_model(model) if model else self.model
        test_loss = model.evaluate(x_test, self.mnist.test.labels)
        print(test_loss)


if __name__ == "__main__":
    lstm_classifier = MnistLSTMClassifier()
    lstm_classifier.train(save_model=True)
    lstm_classifier.evaluate()
    # Load a trained model.
    #lstm_classifier.evaluate(model="./saved_model/lstm-model.h5")
