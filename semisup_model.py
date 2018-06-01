from keras.layers import Input, Dense, LeakyReLU
from keras import regularizers
from keras.Model import Model
from keras import optimizers
from keras.callbacks import ReduceLROnPlateau
from data import x_train, x_val, cats_val


class SemisupNN(object):
    def __init__(self, size, categories=True):
        """
        categories == True means use attack categories, categories == False means use just attack vs not attack
        """
        self.categories = categories

        self.input_layer = Input(shape=(x_train.shape[1],))

        # encoder
        self.encoder = self.get_enc_layer(size, (x_train.shape[1],))
        self.encoded = self.encoder(encoded)

        # decoder
        if categories:
            self.decoded = Dense(10, activation='sigmoid', name='label_output')(self.encoded)
        else:
            self.decoded = Dense(1, activation='sigmoid', name='label_output')(self.encoded)
        self.reconstruct = Dense(x_train.shape[1], activation='sigmoid', name='reconstruct_output')(self.encoded)

    def get_enc_layer(dim, inp_dim):
        inp_layer = Input(shape=inp_dim)
        encodedn = Dense(dim, activation=None, kernel_initializer='glorot_normal',
                         kernel_regularizer=regularizers.l2(0.1),
                         bias_regularizer=regularizers.l2(0.1)
                   )(inp_layer)
        encoded = LeakyReLU(alpha=0.2)(encoded)
        model = Model(inp_layer, encoded)
        return model

    def train(self):
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
            labels = cats_train
            labels_val = cats_val
        else:
            labels = y_train
            labels_val = y_val
        model.fit(x_train, {'label_output': labels_train, 'reconstruct_output': x_train},
                  epochs=500, batch_size=1000, shuffle=False,
                  validation_data=(x_val, [labels_val, x_val])


    def get_embeddings(self, data):
        enc = Model(self.input_layer, self.encoded)
        return enc.predict(data)

