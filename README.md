# Keras LAMB

[![Travis](https://travis-ci.org/CyberZHG/keras-lamb.svg)](https://travis-ci.org/CyberZHG/keras-lamb)
[![Coverage](https://coveralls.io/repos/github/CyberZHG/keras-lamb/badge.svg?branch=master)](https://coveralls.io/github/CyberZHG/keras-lamb)

[LAMB](https://arxiv.org/pdf/1904.00962.pdf) (Layer-wise Adaptive Moments optimizer for Batch training) in Keras.

## Install

```bash
python setup.py install
```

## Usage

```python
from keras.layers import Dense
from keras.models import Sequential
from keras_lamb import Lamb

model = Sequential()
model.add(Dense(input_shape=(5,), units=3))
model.compile(optimizer=Lamb(), loss='mse')
```
