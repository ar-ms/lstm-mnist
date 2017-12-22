### lstm-mnist
Image classification using a RNN classifier(LSTM) with Keras.

### Requirements
   - python3
   - tensorflow (>=1.4)
   - keras (>=2.1.1)
   - numpy

### Train and evaluate
The classifier is trained on 55k samples and tested on 10k samples (The default split).

The ANN is made of one [LSTM layer](https://keras.io/layers/recurrent/#lstm) with 128 hidden units and one [dense](https://keras.io/layers/core/#dense) output layer of 10 units with [softmax activation](https://keras.io/activations/#softmax).
The [rmsprop optimizer](https://keras.io/optimizers/#rmsprop) is used with [categorial_crossentropy](https://keras.io/losses/#categorical_crossentropy) as loss function.

Launch lstm_classifier.py to train and evaluate the classifier, you can dump a trained classifier and load it later.

`python lstm_classifier.py`

### Ressources
  - [Long Short Term Memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf)
  - [Understanding LSTMs](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
  - [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
