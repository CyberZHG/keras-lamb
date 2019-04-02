import os
import tempfile
from unittest import TestCase
import numpy as np
from keras.layers import Dense
from keras.constraints import max_norm
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras_lamb import Lamb


class TestOptimizer(TestCase):

    def test_fit(self):
        model = Sequential()
        model.add(Dense(input_shape=(5,), units=3, bias_constraint=max_norm(10.0)))
        model.compile(optimizer=Lamb(decay=1e-6), loss='mse')
        model_path = os.path.join(tempfile.gettempdir(), 'keras_lamb_%.4f.h5' % np.random.random())
        model.save(model_path)
        model = load_model(model_path, custom_objects={'Lamb': Lamb})
        model.summary()

        target_w = np.random.standard_normal((5, 3))
        target_b = np.random.standard_normal(3)

        def _date_gen(batch_size=32):
            while True:
                x = np.random.standard_normal((batch_size, 5))
                y = np.dot(x, target_w) + target_b
                yield x, y

        model.fit_generator(
            generator=_date_gen(128),
            steps_per_epoch=500,
            validation_data=_date_gen(),
            validation_steps=50,
            epochs=100,
            callbacks=[
                ReduceLROnPlateau(monitor='val_loss', patience=2, min_lr=1e-5, verbose=True),
                EarlyStopping(monitor='val_loss', patience=5, verbose=True),
            ],
        )
        for i, (batch_x, batch_y) in enumerate(_date_gen(batch_size=1)):
            if i > 100:
                break
            predicted = model.predict(batch_x)
            self.assertTrue(
                np.allclose(predicted, batch_y, rtol=0.0, atol=1e-2),
                [i, predicted, batch_y],
            )
