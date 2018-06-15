from keras.layers import Input, Dense, LeakyReLU
from keras import regularizers
from keras.models import Model
from keras import optimizers
from keras.callbacks import ReduceLROnPlateau


class SemisupNN(object):
    def __init__(self, data, size, categories=True):
        """

        Args:
            data (data.Data):
            size (int):
            categories (bool): True means use attack categories, False means use just binary (attack or not attack).
        """
        self.data = data
        self.categories = categories

        self.input_layer = Input(shape=(self.data.x_train.shape[1],))

        # encoder
        self.encoder = self.get_enc_layer(size, (self.data.x_train.shape[1],))
        self.encoded = self.encoder(self.input_layer)

        # decoder
        if categories:
            self.decoded = Dense(10, activation='sigmoid', name='label_output')(self.encoded)
        else:
            self.decoded = Dense(1, activation='sigmoid', name='label_output')(self.encoded)
        self.reconstruct = Dense(self.data.x_train.shape[1], activation='sigmoid', name='reconstruct_output')(
            self.encoded)

    @staticmethod
    def get_enc_layer(dim, inp_dim):
        inp_layer = Input(shape=inp_dim)
        encoded = Dense(dim, activation=None, kernel_initializer='glorot_normal',
                        kernel_regularizer=regularizers.l2(0.1),
                        bias_regularizer=regularizers.l2(0.1)
                        )(inp_layer)
        encoded = LeakyReLU(alpha=0.2)(encoded)
        model = Model(inp_layer, encoded)
        return model

    def train(self, **kwargs):
        model = Model(self.input_layer, outputs=[self.decoded, self.reconstruct])
        adam = optimizers.Adam(lr=1e-3, decay=1e-4)
        model.compile(optimizer=adam,
                      loss={'label_output': 'categorical_crossentropy', 'reconstruct_output': 'binary_crossentropy'},
                      loss_weights={'label_output': 1., 'reconstruct_output': 1e-1},
                      metrics={'label_output': ['accuracy']})

        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5,
                                      patience=5, verbose=1, min_lr=1e-7,
                                      cooldown=5)

        if self.categories:
            labels = self.data.cats_train
            labels_val = self.data.cats_val
        else:
            labels = self.data.y_train
            labels_val = self.data.y_val

        default_kwargs = {
            'x': self.data.x_train,
            'y': {'label_output': labels, 'reconstruct_output': self.data.x_train},
            'epochs': 500,
            'batch_size': 1000,
            'shuffle': False,
            'validation_data': (self.data.x_val, [labels_val, self.data.x_val]),
            'callbacks': [reduce_lr]
        }
        default_kwargs.update(**kwargs)
        model.fit(**default_kwargs)

    def get_embeddings(self, data):
        enc = Model(self.input_layer, self.encoded)
        return enc.predict(data)
