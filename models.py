import os, errno
import numpy as np
from keras.layers import Input, Dense, LeakyReLU
from keras import regularizers
from keras.models import Model, load_model
from keras import optimizers
from keras.callbacks import ReduceLROnPlateau
from sklearn import decomposition, manifold, discriminant_analysis


class BaseModel(object):
    def __init__(self, data, size):
        self.data = data
        self.size = size

    def train(self, *args, **kwargs):
        raise NotImplementedError

    def get_embeddings(self, data):
        raise NotImplementedError

    def save_model(self, path):
        raise NotImplementedError

    def load_model(self, path):
        raise NotImplementedError

    def get_feature_weights(self):
        raise NotImplementedError


class SemisupNN(BaseModel):
    def __init__(self, data, size, categories=True, reconstruct_weight=0.1,
                 enc_regularizer_weight=0.1, lr=1e-3, lr_decay=1e-5, dec_regularizer_weight=0.,
                 reconstruct_loss='binary_crossentropy', encoder_regularizer='l2'):
        """

        Args:
            data (data.Data):
            size (int):
            categories (bool): True means use attack categories, False means use just binary (attack or not attack).
        """
        super().__init__(data=data, size=size)
        self.categories = categories
        self.reconstruct_weight = reconstruct_weight
        self.enc_regularizer_weight = enc_regularizer_weight
        self.lr = lr
        self.lr_decay = lr_decay
        self.dec_regularizer_weight = dec_regularizer_weight
        self.reconstruct_loss = reconstruct_loss
        if encoder_regularizer == 'l2':
            self.encoder_regularizer = regularizers.l2
        elif encoder_regularizer == 'l1':
            self.encoder_regularizer = regularizers.l1
        else:
            raise ValueError('Invalid encoder_regularizer parameter. Possible are l2,l1.')

        self.input_layer = Input(shape=(self.data.x_train.shape[1],))

        # encoder
        self.encoder = self.get_enc_layer(size, (self.data.x_train.shape[1],))
        self.encoded = self.encoder(self.input_layer)

        # decoder
        if categories:
            self.label_decode = Dense(10, activation='sigmoid', name='label_output',
                                      kernel_regularizer=regularizers.l2(self.dec_regularizer_weight),
                                      bias_regularizer=regularizers.l2(self.dec_regularizer_weight))(self.encoded)
        else:
            self.label_decode = Dense(1, activation='sigmoid', name='label_output',
                                      kernel_regularizer=regularizers.l2(self.dec_regularizer_weight),
                                      bias_regularizer=regularizers.l2(self.dec_regularizer_weight))(self.encoded)
        self.reconstruct = Dense(self.data.x_train.shape[1], activation='sigmoid', name='reconstruct_output')(
            self.encoded)

        self.trainable_model, self.embeddings_model, self.labels_model, self.model_paths = self.make_models()

    def make_models(self):
        trainable_model = Model(self.input_layer, outputs=[self.label_decode, self.reconstruct])
        embeddings_model = Model(self.input_layer, self.encoded)
        labels_model = Model(self.input_layer, self.label_decode)
        model_paths = {
            'trainable_model': 'trainable_model.h5',
            'embeddings_model': 'embeddings_model.h5',
            'labels_model': 'labels_model.h5'
        }

        return trainable_model, embeddings_model, labels_model, model_paths

    def save_model(self, path):
        try:
            os.makedirs(path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise ValueError('Path for saving model already exists!')
        for model in self.model_paths:
            fname = self.model_paths[model]
            if fname is not None:
                self.__dict__[model].save(os.path.join(path, fname))

    def load_model(self, path):
        for model in self.model_paths:
            fname = self.model_paths[model]
            if fname is None:
                self.__dict__[model] = None
            else:
                self.__dict__[model] = load_model(os.path.join(path, fname))

    def get_enc_layer(self, dim, inp_dim):
        inp_layer = Input(shape=inp_dim)
        encoded = Dense(dim, activation=None, kernel_initializer='glorot_normal',
                        kernel_regularizer=self.encoder_regularizer(self.enc_regularizer_weight),
                        bias_regularizer=self.encoder_regularizer(self.enc_regularizer_weight)
                        )(inp_layer)
        encoded = LeakyReLU(alpha=0.2)(encoded)
        model = Model(inp_layer, encoded)
        return model

    def get_feature_weights(self):
        """Return weights of the encoder."""
        return self.encoder.get_weights()[0]

    def train(self, **kwargs):
        model = self.trainable_model
        adam = optimizers.Adam(lr=self.lr, decay=self.lr_decay)
        if self.categories:
            label_loss = 'categorical_crossentropy'
        else:
            label_loss = 'binary_crossentropy'
        model.compile(optimizer=adam,
                      loss={'label_output': label_loss, 'reconstruct_output': self.reconstruct_loss},
                      loss_weights={'label_output': 1., 'reconstruct_output': self.reconstruct_weight},
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
            'epochs': 1500,
            'batch_size': 1000,
            'shuffle': False,
            'validation_data': (self.data.x_val, [labels_val, self.data.x_val]),
            'callbacks': [reduce_lr]
        }
        default_kwargs.update(**kwargs)
        return model.fit(**default_kwargs)

    def get_embeddings(self, data):
        enc = self.embeddings_model
        return enc.predict(data)

    def predict_labels(self, data):
        labels = self.labels_model
        return labels.predict(data)


class UnsupNN(SemisupNN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.label_decode
        del self.categories

    def make_models(self):
        trainable_model = Model(self.input_layer, self.reconstruct)
        embeddings_model = Model(self.input_layer, self.encoded)
        labels_model = None
        model_paths = {
            'trainable_model': 'trainable_model.h5',
            'embeddings_model': 'embeddings_model.h5',
            'labels_model': None
        }

        return trainable_model, embeddings_model, labels_model, model_paths

    def train(self, **kwargs):
        model = self.trainable_model
        adam = optimizers.Adam(lr=self.lr, decay=self.lr_decay)
        model.compile(optimizer=adam,
                      loss=self.reconstruct_loss)

        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5,
                                      patience=5, verbose=1, min_lr=1e-7,
                                      cooldown=5)

        default_kwargs = {
            'x': self.data.x_train,
            'y': self.data.x_train,
            'epochs': 1500,
            'batch_size': 1000,
            'shuffle': False,
            'validation_data': (self.data.x_val, self.data.x_val),
            'callbacks': [reduce_lr],
        }
        default_kwargs.update(**kwargs)
        return model.fit(**default_kwargs)


class SupNN(SemisupNN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.reconstruct

    def make_models(self):
        trainable_model = Model(self.input_layer, self.label_decode)
        embeddings_model = None
        labels_model = Model(self.input_layer, self.label_decode)
        model_paths = {
            'trainable_model': 'trainable_model.h5',
            'embeddings_model': None,
            'labels_model': 'labels_model.h5'
        }

        return trainable_model, embeddings_model, labels_model, model_paths

    def train(self, **kwargs):
        model = self.trainable_model
        adam = optimizers.Adam(lr=self.lr, decay=self.lr_decay)
        if self.categories:
            label_loss = 'categorical_crossentropy'
        else:
            label_loss = 'binary_crossentropy'
        model.compile(optimizer=adam,
                      loss=label_loss)

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
            'y': labels,
            'epochs': 1500,
            'batch_size': 1000,
            'shuffle': False,
            'validation_data': (self.data.x_val, labels_val),
            'callbacks': [reduce_lr],
        }
        default_kwargs.update(**kwargs)
        return model.fit(**default_kwargs)


class PCA(BaseModel):
    def __init__(self, data, size):
        super().__init__(data=data, size=size)
        self.pca = decomposition.PCA()

    def train(self):
        return self.pca.fit(self.data.x_train)

    def get_embeddings(self, data):
        return self.pca.transform(data)[:, :self.size]

    def get_feature_weights(self):
        return self.pca.components_


class LDA(BaseModel):
    def __init__(self, data, size, categories=False):
        super().__init__(data=data, size=size)
        self.categories = categories
        self.lda = discriminant_analysis.LinearDiscriminantAnalysis(n_components=size)

    def train(self, *args, **kwargs):
        if self.categories:
            return self.lda.fit(self.data.x_train, self.data.cats_train)
        else:
            return self.lda.fit(self.data.x_train, self.data.y_train)

    def get_embeddings(self, data):
        return self.lda.transform(data)

    def get_feature_weights(self):
        return self.lda.scalings_[:,:self.size]


class TSNE(BaseModel):
    def __init__(self, data, size, *args, **kwargs):
        super().__init__(data=data, size=size)
        self.tsne = manifold.TSNE(*args, **kwargs)
        self.all_data = np.concatenate((self.data.x_train, self.data.x_test))
        self.all_data_transformed = None

    def train(self):
        self.all_data_transformed = self.tsne.fit_transform(self.all_data)
        return self.tsne


    def get_embeddings(self, data):
        """
        Args:
            data: assumed one of train data or test data

        Returns:
            corresponding data to input, transformed by t-SNE
        """
        # check if input is train or test data... assumes train is bigger than test
        if len(data) > len(self.all_data_transformed) / 2:
            return self.all_data_transformed[:self.data.x_train.shape[0]]  # get only train data
        else:
            return self.all_data_transformed[self.data.x_train.shape[0]:]  # get only test data


class MDS(BaseModel):
    def __init__(self, data, size, *args, **kwargs):
        super().__init__(data=data, size=size)
        self.mds = manifold.MDS(*args, **kwargs)
        self.all_data = np.concatenate((self.data.x_train, self.data.x_test))
        self.all_data_transformed = None

    def train(self):
        self.all_data_transformed = self.mds.fit_transform(self.all_data)
        return self.mds

    def get_embeddings(self, data):
        """
        Args:
            data: assumed one of train data or test data

        Returns:
            corresponding data to input, transformed by t-SNE
        """
        # check if input is train or test data... assumes train is bigger than test
        if len(data) > len(self.all_data_transformed) / 2:
            return self.all_data_transformed[:self.data.x_train.shape[0]]  # get only train data
        else:
            return self.all_data_transformed[self.data.x_train.shape[0]:]  # get only test data
