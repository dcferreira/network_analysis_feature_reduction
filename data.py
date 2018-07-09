import random
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


random.seed(1337)
np.random.seed(1337)
tf.set_random_seed(1337)


class Data(object):
    def __init__(self, all, training, testing):
        big_df = read_df(all)
        unsup = big_df
        train = read_df(training, big_df)
        test = read_df(testing, big_df)
        self.columns = test.columns

        # split train/dev/test
        unsup_mat = unsup.as_matrix()
        x_train, x_val, y_train, y_val, cats_train, cats_val = \
            train_test_split(train.as_matrix()[:,:-11], # feats
                             train.as_matrix()[:,-1], # labels
                             train.as_matrix()[:,-11:-1], # cats
                             test_size=0.2, random_state=1337)
        x_test, y_test, cats_test = (test.as_matrix()[:,:-11],
                                     test.as_matrix()[:,-1],
                                     test.as_matrix()[:,-11:-1])

        # normalize
        maximum = unsup_mat[:,:-11].max(axis=0)
        minimum = unsup_mat[:,:-11].min(axis=0)

        unsup_mat[:,:-11] = (unsup_mat[:,:-11] - minimum) / (maximum - minimum)
        self.x_train = (x_train - minimum) / (maximum - minimum)
        self.x_val = (x_val - minimum) / (maximum - minimum)
        self.x_test = (x_test - minimum) / (maximum - minimum)
        self.y_train, self.y_test, self.y_val = y_train, y_test, y_val
        self.cats_nr_train, self.cats_nr_test, self.cats_nr_val = (
            [np.argmax(x) for x in cats_train],
            [np.argmax(x) for x in cats_test],
            [np.argmax(x) for x in cats_val]
        )
        self.cats_train, self.cats_test, self.cats_val = cats_train, cats_test, cats_val


def read_df(filename, big_df=None, onehot=True, cat_only=False, num_only=False):
    categorical_feats = ['proto', 'state', 'service', 'is_sm_ips_ports',
                         'is_ftp_login', 'attack_cat', 'label']
    all_feats = ["proto","state","dur","sbytes","dbytes",
                 "sttl","dttl","sloss","dloss","service",
                 "sload","dload","spkts","dpkts","swin","dwin",
                 "stcpb","dtcpb","smean","dmean","trans_depth",
                 "response_body_len","sjit","djit","sinpkt","dinpkt",
                 "tcprtt","synack","ackdat","is_sm_ips_ports",
                 "ct_state_ttl","ct_flw_http_mthd","is_ftp_login",
                 "ct_ftp_cmd","ct_srv_src","ct_srv_dst",
                 "ct_dst_ltm","ct_src_ltm","ct_src_dport_ltm",
                 "ct_dst_sport_ltm","ct_dst_src_ltm",
                 "attack_cat","label"]
    
    df = pd.read_csv(filename,
                     index_col=False,
                     dtype={f: 'str' for f in categorical_feats},
                     usecols=all_feats
                    )
    
    # replace attack cat names' with correct versions
    df.attack_cat.replace('Backdoors', 'Backdoor', inplace=True)
    df.attack_cat.replace([' Fuzzers', ' Fuzzers '], 'Fuzzers', inplace=True)
    df.attack_cat.replace(' Reconnaissance ', 'Reconnaissance', inplace=True)
    df.attack_cat.replace(' Shellcode ', 'Shellcode', inplace=True)
    
    # add "rate" column
    df['rate'] = ((df.spkts + df.dpkts - 1) / df.dur).replace([np.inf], 0)
    
    # replace empty strings with NaN
    df.ct_ftp_cmd = df.ct_ftp_cmd.replace('\s+', np.nan, regex=True).astype('float')
    
    if cat_only:
        df = df.select_dtypes(exclude=np.number)
    elif num_only:
        new_df = df.select_dtypes(include=np.number)
        new_df['label'] = df.label
        df = new_df

    if onehot:
        ddf = pd.get_dummies(df)
        if big_df is not None:
            dummies_frame = pd.get_dummies(big_df)
            ddf = ddf.reindex(columns=dummies_frame.columns, fill_value=0)
    else:
        ddf = df
    return ddf.fillna(value=0)

