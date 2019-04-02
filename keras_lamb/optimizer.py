""" Copyright 2019 Zhao HG
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from keras.legacy import interfaces
from keras.optimizers import Optimizer
import keras.backend as K

__all__ = ['Lamb']


class Lamb(Optimizer):
    """LAMP optimizer.

    Default parameters follow those provided in the original paper.

    # Arguments
        lr: float >= 0. Learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
        decay: float >= 0. Learning rate decay over each update.
        weight_decay: float >= 0. Weight decay rate.
        lower_bound: float >= 0. Minimum value of trust ratios.
        upper_bound: float >= 0. Maximum value of trust ratios.

    # References
        - [Reducing BERT Pre-Training Time from 3 Days to 76 Minutes]
          (https://arxiv.org/abs/1904.00962)
    """
    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-6, decay=0., weight_decay=0.01,
                 lower_bound=1e-3, upper_bound=10.0, **kwargs):
        super(Lamb, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(decay, name='decay')
            self.weight_decay = K.variable(weight_decay, name='weight_decay')
        self.epsilon = epsilon
        self.initial_decay = decay
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        t = K.cast(self.iterations, K.floatx()) + 1
        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * t))

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        self.weights = [self.iterations] + ms + vs

        for p, g, m, v in zip(params, grads, ms, vs):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))

            mhat_t = m_t / (1. - K.pow(self.beta_1, t))
            vhat_t = v_t / (1. - K.pow(self.beta_2, t))

            u_t = mhat_t / K.sqrt(vhat_t + self.epsilon) + self.weight_decay * p
            trust_ratio = K.sqrt(K.sum(K.square(p)) / K.sum(K.square(u_t)))
            trust_ratio = K.minimum(K.maximum(trust_ratio, self.lower_bound), self.upper_bound)

            lr_p = trust_ratio * lr
            new_p = p - lr_p * u_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'decay': float(K.get_value(self.decay)),
                  'weight_decay': float(K.get_value(self.weight_decay)),
                  'epsilon': self.epsilon,
                  'upper_bound': self.upper_bound,
                  'lower_bound': self.lower_bound}
        base_config = super(Lamb, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
