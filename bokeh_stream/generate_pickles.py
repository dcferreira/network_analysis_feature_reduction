import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
try:
    from models import SemisupNN
    from data import Data
except ImportError:
    raise ImportError('Error importing! Add the root of the repository to your PYTHONPATH and try again.')


AE_PARAM = '1e-1'
AE_MODEL_PATH = '../experiments/cats_ae/run_mse_recweights_%s/0' % AE_PARAM

data = Data('../UNSW-NB15_all.csv', '../UNSW_NB15_training-set.csv', '../UNSW_NB15_testing-set.csv')
cats_ae = SemisupNN(data, 2, categories=True, reconstruct_weight=float(AE_PARAM), reconstruct_loss='mse')
cats_ae.load_model(AE_MODEL_PATH)

cats_ae_classifier = DecisionTreeClassifier()
original_classifier = DecisionTreeClassifier()

train_embs = cats_ae.get_embeddings(data.x_train)
test_embs = cats_ae.get_embeddings(data.x_test)

cats_ae_classifier.fit(train_embs, data.y_train)
original_classifier.fit(data.x_train, data.y_train)

test_x_enc = test_embs
cats_ae_pred_y = cats_ae_classifier.predict(test_x_enc)

original_pred_y = original_classifier.predict(data.x_test)


# scale projections (to be able to plot them together)
def scale(*arrays):
    minimum = np.min([np.min(ar, axis=0) for ar in arrays], axis=0)
    maximum = np.max([np.max(ar, axis=0) for ar in arrays], axis=0)
    return ((ar - minimum) / (maximum - minimum) for ar in arrays)


test_x_enc_norm, train_x_enc_norm = scale(test_x_enc, train_embs)

original_pred_y = np.array(original_pred_y, dtype=int)
cats_ae_pred_y = np.array(cats_ae_pred_y, dtype=int)

df = pd.DataFrame({'original_pred': original_pred_y,
                   'cats_ae_pred': cats_ae_pred_y,
                   'x_cats_ae': test_x_enc_norm[:, 0], 'y_cats_ae': test_x_enc_norm[:, 1],
                   'label': data.y_test, 'category': np.array(data.cats_nr_test, dtype=int),
                   'cat_str': np.array(data.cats_nr_test, dtype=str),
                   'cats_ae_pred_str': np.array(cats_ae_pred_y, dtype=str),
                   'original_pred_str': np.array(original_pred_y, dtype=str)},
                  index=np.arange(0, len(original_pred_y)))
df.to_pickle('dataframe.pkl')
np.save('cats_ae_x_train_scaled', train_x_enc_norm, allow_pickle=True)
np.save('cats_nr_train', np.array(data.cats_nr_train, dtype=int), allow_pickle=True)
print(data.columns[-11:-1])
