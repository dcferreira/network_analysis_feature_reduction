import errno
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
from keras import optimizers
from keras import regularizers
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Input, Dense, LeakyReLU
from keras.models import Model, load_model
from sklearn import decomposition, manifold, discriminant_analysis
from torch import nn


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
            self.label_decode = Dense(10, activation='softmax', name='label_output',
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
            return self.lda.fit(self.data.x_train, self.data.cats_nr_train)
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
        self.all_data = np.concatenate((self.data.x_train, self.data.x_val, self.data.x_test))
        self.all_data_transformed = None

    def train(self):
        self.all_data_transformed = self.tsne.fit_transform(self.all_data)
        return self.tsne

    def save_model(self, path):
        try:
            os.makedirs(path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise ValueError('Path for saving model already exists!')
        np.save(os.path.join(path, 'transformed_data.npy'), self.all_data_transformed)

    def load_model(self, path):
        self.all_data_transformed = np.load(os.path.join(path, 'transformed_data.npy'))

    def get_embeddings(self, data):
        """
        Args:
            data: assumed one of train data or test data

        Returns:
            corresponding data to input, transformed by t-SNE
        """
        # check if input is train or test/val data... assumes train is bigger than test and val
        if len(data) == len(self.data.x_train):
            return self.all_data_transformed[:self.data.x_train.shape[0]]
        elif len(data) == len(self.data.x_val):
            return self.all_data_transformed[self.data.x_train.shape[0]:self.data.x_train.shape[0] +
                                                                        self.data.x_val.shape[0]]
        elif len(data) == len(self.data.x_test):
            return self.all_data_transformed[self.data.x_train.shape[0] + self.data.x_val.shape[0]:]
        else:
            raise ValueError('T-SNE can only return the train, test or validation sets embeddings!')


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


class DeepEncoderModel(nn.Module):
    def __init__(self, dim, inp_dim):
        super().__init__()
        self.fc1 = nn.Linear(inp_dim, 20)
        self.bn1 = nn.BatchNorm1d(20)
        self.fc2 = nn.Linear(20, 10)
        self.bn2 = nn.BatchNorm1d(10)
        self.dense = nn.Linear(10, dim)
        torch.nn.init.xavier_normal_(self.dense.weight)
        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = self.bn1(x)
        x = F.leaky_relu(self.fc2(x))
        x = self.bn2(x)
        x = F.leaky_relu(self.dense(x))
        return x


class DeepDecoderModel(nn.Module):
    def __init__(self, dim, inp_dim, categories):
        super().__init__()
        self.fc1 = nn.Linear(dim, 10)
        self.bn1 = nn.BatchNorm1d(10)
        self.fc2 = nn.Linear(10, 20)
        self.bn2 = nn.BatchNorm1d(20)
        if categories:
            self.dense = nn.Linear(20, 10)
            self.act = nn.Softmax(dim=1)
        else:
            self.dense = nn.Linear(20, 1)
            self.act = nn.Sigmoid()
        self.reconstruct = nn.Linear(20, inp_dim)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = self.bn1(x)
        x = F.leaky_relu(self.fc2(x))
        x = self.bn2(x)
        label = self.act(self.dense(x))
        reconstruct = nn.Sigmoid()(self.reconstruct(x))

        return label, reconstruct


class DeepSemiSupNN(BaseModel):
    def __init__(self, data, size, categories=True, reconstruct_weight=0.1,
                 lr=1e-3, lr_decay=1e-5, reconstruct_loss='binary_crossentropy', random_seed=1337,
                 checkpoint_path='tmp/'):
        """

        Args:
            data (data.Data):
            size (int):
            categories (bool): True means use attack categories, False means use just binary (attack or not attack).
        """
        super().__init__(data=data, size=size)
        self.categories = categories
        self.reconstruct_weight = reconstruct_weight
        self.lr = lr
        self.lr_decay = lr_decay
        if reconstruct_loss == 'binary_crossentropy':
            self.reconstruct_loss = F.binary_cross_entropy
        elif reconstruct_loss == 'mse':
            self.reconstruct_loss = F.mse_loss
        else:
            raise ValueError(f'Unknown reconstruct loss: `{reconstruct_loss}`. '
                             f'Please use `mse` or `binary_crossentropy`')
        self.size = size
        self.checkpoint_path = checkpoint_path

        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

        self.encoder = self.get_encoder()
        self.decoder = self.get_decoder()

        params = list(self.encoder.parameters())
        params.extend(list(self.decoder.parameters()))
        self.optimizer = torch.optim.Adam(params, lr=self.lr, weight_decay=self.lr_decay)

    def get_encoder(self):
        return DeepEncoderModel(self.size, self.data.x_train.shape[1])

    def get_decoder(self):
        return DeepDecoderModel(self.size, self.data.x_train.shape[1], self.categories)

    def select_labels(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        if self.categories:
            selection = np.linalg.norm(y_true, axis=1) > 1e-6  # if belongs to some class
            selection = torch.ByteTensor(selection.astype('uint8'))
        else:
            selection = y_true > -1
        selected_pred = y_pred[selection]
        selected_true = y_true[selection]

        return selected_pred, selected_true

    def loss(self, y_pred: torch.Tensor, y_true: torch.Tensor, reconstruct: torch.Tensor,
             input_data: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        selected_pred, selected_true = self.select_labels(y_pred, y_true)
        if self.categories:
            label_loss = F.cross_entropy(selected_pred,  # predictions are arrays of size #n_class
                                         selected_true.max(1)[1])
        else:
            label_loss = F.binary_cross_entropy(selected_pred, selected_true)

        reconstruction_loss = self.reconstruct_loss(reconstruct, input_data)

        return label_loss + self.reconstruct_weight * reconstruction_loss, label_loss, reconstruction_loss

    def run_step(self, data: torch.Tensor, labels: torch.Tensor) -> (torch.Tensor, torch.Tensor,
                                                                     torch.Tensor, torch.Tensor):
        y_pred, reconstruct = self.decoder(self.encoder(data))
        total_loss, label_loss, reconstruction_loss = self.loss(y_pred, labels, reconstruct, data)
        return y_pred, total_loss, label_loss, reconstruction_loss

    def train_step(self, data, labels) -> (float, float, float, torch.Tensor):
        self.encoder.train()
        self.decoder.train()

        self.optimizer.zero_grad()
        y_pred, total_loss, label_loss, reconstruction_loss = self.run_step(data, labels)
        total_loss.backward()
        self.optimizer.step()
        return total_loss.item(), label_loss.item(), reconstruction_loss.item(), y_pred

    def count_predictions(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> (int, int):
        selected_pred, selected_true = self.select_labels(y_pred, y_true)
        if self.categories:
            predicted_labels = np.argmax(selected_pred.detach().numpy(), axis=1)
            true_nr = np.argmax(selected_true.numpy(), axis=1)
        else:
            predicted_labels = np.round(selected_pred.view(-1).detach().numpy())
            true_nr = selected_true.numpy()
        # noinspection PyTypeChecker
        right = sum(predicted_labels == true_nr)
        total = len(selected_pred)
        return right, total

    def train(self, n_epochs=300, verbose=True):
        train_data = torch.utils.data.TensorDataset(
            torch.Tensor(self.data.x_train),
            torch.Tensor(self.data.cats_train) if self.categories else torch.Tensor(self.data.y_train)
        )
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=1000, shuffle=False)

        val_data = torch.utils.data.TensorDataset(
            torch.Tensor(self.data.x_val),
            torch.Tensor(self.data.cats_val) if self.categories else torch.Tensor(self.data.y_val)
        )
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=1000, shuffle=False)

        best_val_accuracy = -1
        best_epoch = None
        for epoch in range(n_epochs):
            total_loss = 0
            label_loss = 0
            reconstruction_loss = 0
            corrects = 0
            labeled_total = 0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                tot_loss, lab_loss, rec_loss, y_pred = self.train_step(inputs, labels)
                label_loss += lab_loss
                reconstruction_loss += rec_loss
                total_loss += tot_loss

                c, tot = self.count_predictions(y_pred, labels)
                corrects += c
                labeled_total += tot

            val_total_loss, val_label_loss, val_reconstruction_loss, val_accuracy = self.evaluate(val_loader)
            if verbose:
                val_norm = np.linalg.norm(self.get_embeddings(self.data.x_val), axis=1)
                print(f'Epoch {epoch}/{n_epochs}\n\t'
                      f'loss: {total_loss:.4f} - '
                      f'label_output_loss: {label_loss:.4f} - '
                      f'reconstruct_output_loss: {reconstruction_loss:.4f} - '
                      f'label_output_acc: {corrects / labeled_total:.4f} - '
                      f'val_loss: {val_total_loss:.4f} - '
                      f'val_label_output_loss: {val_label_loss:.4f} - '
                      f'val_reconstruct_output_loss: {val_reconstruction_loss:.4f} - '
                      f'val_output_acc: {val_accuracy:.4f} - '
                      f'val_norm_mean: {val_norm.mean()} +- {val_norm.std()}')

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_epoch = epoch
                self.save_model(self.checkpoint_path)

        if best_epoch is not None:
            print(f'\nFinished training, restoring checkpoint from epoch {best_epoch}...')
            self.load_model(self.checkpoint_path)

    def evaluate(self, val_loader: torch.utils.data.DataLoader) -> (float, float, float, float):
        self.encoder.eval()
        self.decoder.eval()

        total_loss = 0
        label_loss = 0
        reconstruction_loss = 0
        corrects = 0
        labeled_total = 0
        for i, data in enumerate(val_loader, 0):
            inputs, labels = data
            y_pred, tot_loss, lab_loss, rec_loss = self.run_step(inputs, labels)
            label_loss += lab_loss.item()
            reconstruction_loss += rec_loss.item()
            total_loss += tot_loss.item()

            c, tot = self.count_predictions(y_pred, labels)
            corrects += c
            labeled_total += tot

        return total_loss, label_loss, reconstruction_loss, corrects / labeled_total

    def get_embeddings(self, data):
        self.encoder.eval()
        data_tensor = torch.Tensor(data)
        output = self.encoder(data_tensor)
        return output.detach().numpy()

    def predict_labels(self, data):
        self.encoder.eval()
        self.decoder.eval()
        data_tensor = torch.Tensor(data)
        labels, _ = self.decoder(self.encoder(data_tensor))
        return labels.detach().numpy()

    def save_model(self, path):
        try:
            os.makedirs(path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise ValueError('Path for saving model already exists!')
        torch.save(self.encoder.state_dict(), os.path.join(path, 'encoder.pth'))
        torch.save(self.decoder.state_dict(), os.path.join(path, 'decoder.pth'))

    def load_model(self, path):
        self.encoder.load_state_dict(torch.load(os.path.join(path, 'encoder.pth')))
        self.decoder.load_state_dict(torch.load(os.path.join(path, 'decoder.pth')))

    def get_feature_weights(self):
        raise NotImplementedError
