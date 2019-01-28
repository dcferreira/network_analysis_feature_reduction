import random
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

random.seed(1337)
np.random.seed(1337)
tf.set_random_seed(1337)


class Data(ABC):
    @property
    @abstractmethod
    def x_train(self):
        pass

    @property
    @abstractmethod
    def x_val(self):
        pass

    @property
    @abstractmethod
    def x_test(self):
        pass

    @property
    @abstractmethod
    def y_train(self):
        pass

    @property
    @abstractmethod
    def y_val(self):
        pass

    @property
    @abstractmethod
    def y_test(self):
        pass

    @property
    @abstractmethod
    def cats_train(self):
        pass

    @property
    @abstractmethod
    def cats_val(self):
        pass

    @property
    @abstractmethod
    def cats_test(self):
        pass


class UNSW15Data(Data):
    def __init__(self, all_data, training, testing):
        big_df = read_unsw_df(all_data)
        unsup = big_df
        train_df = read_unsw_df(training, big_df)
        test_df = read_unsw_df(testing, big_df)
        self.columns = train_df.columns
        train = train_df.values
        test = test_df.values

        # use the full train set with classifiers (they use crossvalidation)
        x_full_train, self.y_full_train, self.cats_full_train = (train[:, :-11],
                                                                 train[:, -1],
                                                                 train[:, -11:-1])

        # split train/dev/test
        unsup_mat = unsup.values
        x_train, x_val, y_train, y_val, cats_train, cats_val = \
            train_test_split(train[:, :-11],  # feats
                             train[:, -1],  # labels
                             train[:, -11:-1],  # cats
                             test_size=0.2, random_state=1337)
        x_test, y_test, cats_test = (test[:, :-11],
                                     test[:, -1],
                                     test[:, -11:-1])

        # normalize
        maximum = unsup_mat[:, :-11].max(axis=0)
        minimum = unsup_mat[:, :-11].min(axis=0)

        # unsup_mat[:, :-11] = (unsup_mat[:, :-11] - minimum) / (maximum - minimum)
        self.x_full_train = (x_full_train - minimum) / (maximum - minimum)
        self._x_train = (x_train - minimum) / (maximum - minimum)
        self._x_val = (x_val - minimum) / (maximum - minimum)
        self._x_test = (x_test - minimum) / (maximum - minimum)
        self._y_train, self._y_test, self._y_val = y_train, y_test, y_val
        self.cats_nr_full_train, self.cats_nr_train, self.cats_nr_test, self.cats_nr_val = (
            [np.argmax(x) if np.sum(x) > 1e-6 else -1 for x in self.cats_full_train],
            [np.argmax(x) if np.sum(x) > 1e-6 else -1 for x in cats_train],
            [np.argmax(x) if np.sum(x) > 1e-6 else -1 for x in cats_test],
            [np.argmax(x) if np.sum(x) > 1e-6 else -1 for x in cats_val]
        )
        self._cats_train, self._cats_test, self._cats_val = cats_train, cats_test, cats_val

    @property
    def x_train(self):
        return self._x_train

    @property
    def x_val(self):
        return self._x_val

    @property
    def x_test(self):
        return self._x_test

    @property
    def y_train(self):
        return self._y_train

    @property
    def y_val(self):
        return self._y_val

    @property
    def y_test(self):
        return self._y_test

    @property
    def cats_train(self):
        return self._cats_train

    @property
    def cats_val(self):
        return self._cats_val

    @property
    def cats_test(self):
        return self._cats_test


def read_unsw_df(filename, big_df=None, onehot=True, cat_only=False, num_only=False):
    categorical_feats = ['proto', 'state', 'service', 'is_sm_ips_ports',
                         'is_ftp_login', 'attack_cat', 'label']
    all_feats = ["proto", "state", "dur", "sbytes", "dbytes",
                 "sttl", "dttl", "sloss", "dloss", "service",
                 "sload", "dload", "spkts", "dpkts", "swin", "dwin",
                 "stcpb", "dtcpb", "smean", "dmean", "trans_depth",
                 "response_body_len", "sjit", "djit", "sinpkt", "dinpkt",
                 "tcprtt", "synack", "ackdat", "is_sm_ips_ports",
                 "ct_state_ttl", "ct_flw_http_mthd", "is_ftp_login",
                 "ct_ftp_cmd", "ct_srv_src", "ct_srv_dst",
                 "ct_dst_ltm", "ct_src_ltm", "ct_src_dport_ltm",
                 "ct_dst_sport_ltm", "ct_dst_src_ltm",
                 "attack_cat", "label"]

    dtypes = {f: 'str' for f in categorical_feats}
    dtypes['label'] = 'int'
    df = pd.read_csv(filename,
                     index_col=False,
                     dtype=dtypes,
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
            for v in list(ddf):
                assert v in dummy_variable_list, v
        ddf = ddf.reindex(columns=dummy_variable_list, fill_value=0)
    else:
        ddf = df
    return ddf.fillna(value=0)


dummy_variable_list = ['dur', 'sbytes', 'dbytes', 'sttl', 'dttl', 'sloss', 'dloss', 'sload', 'dload', 'spkts', 'dpkts',
                       'swin', 'dwin', 'stcpb', 'dtcpb', 'smean', 'dmean', 'trans_depth', 'response_body_len', 'sjit',
                       'djit', 'sinpkt', 'dinpkt', 'tcprtt', 'synack', 'ackdat', 'ct_state_ttl', 'ct_flw_http_mthd',
                       'ct_ftp_cmd', 'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ltm', 'ct_src_dport_ltm',
                       'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'rate', 'proto_3pc', 'proto_a/n',
                       'proto_aes-sp3-d', 'proto_any', 'proto_argus', 'proto_aris', 'proto_arp', 'proto_ax.25',
                       'proto_bbn-rcc', 'proto_bna', 'proto_br-sat-mon', 'proto_cbt', 'proto_cftp', 'proto_chaos',
                       'proto_compaq-peer', 'proto_cphb', 'proto_cpnx', 'proto_crtp', 'proto_crudp', 'proto_dcn',
                       'proto_ddp', 'proto_ddx', 'proto_dgp', 'proto_egp', 'proto_eigrp', 'proto_emcon', 'proto_encap',
                       'proto_esp', 'proto_etherip', 'proto_fc', 'proto_fire', 'proto_ggp', 'proto_gmtp', 'proto_gre',
                       'proto_hmp', 'proto_i-nlsp', 'proto_iatp', 'proto_ib', 'proto_icmp', 'proto_idpr',
                       'proto_idpr-cmtp', 'proto_idrp', 'proto_ifmp', 'proto_igmp', 'proto_igp', 'proto_il', 'proto_ip',
                       'proto_ipcomp', 'proto_ipcv', 'proto_ipip', 'proto_iplt', 'proto_ipnip', 'proto_ippc',
                       'proto_ipv6', 'proto_ipv6-frag', 'proto_ipv6-no', 'proto_ipv6-opts', 'proto_ipv6-route',
                       'proto_ipx-n-ip', 'proto_irtp', 'proto_isis', 'proto_iso-ip', 'proto_iso-tp4', 'proto_kryptolan',
                       'proto_l2tp', 'proto_larp', 'proto_leaf-1', 'proto_leaf-2', 'proto_merit-inp', 'proto_mfe-nsp',
                       'proto_mhrp', 'proto_micp', 'proto_mobile', 'proto_mtp', 'proto_mux', 'proto_narp',
                       'proto_netblt', 'proto_nsfnet-igp', 'proto_nvp', 'proto_ospf', 'proto_pgm', 'proto_pim',
                       'proto_pipe', 'proto_pnni', 'proto_pri-enc', 'proto_prm', 'proto_ptp', 'proto_pup', 'proto_pvp',
                       'proto_qnx', 'proto_rdp', 'proto_rsvp', 'proto_rtp', 'proto_rvd', 'proto_sat-expak',
                       'proto_sat-mon', 'proto_sccopmce', 'proto_scps', 'proto_sctp', 'proto_sdrp', 'proto_secure-vmtp',
                       'proto_sep', 'proto_skip', 'proto_sm', 'proto_smp', 'proto_snp', 'proto_sprite-rpc', 'proto_sps',
                       'proto_srp', 'proto_st2', 'proto_stp', 'proto_sun-nd', 'proto_swipe', 'proto_tcf', 'proto_tcp',
                       'proto_tlsp', 'proto_tp++', 'proto_trunk-1', 'proto_trunk-2', 'proto_ttp', 'proto_udp',
                       'proto_udt', 'proto_unas', 'proto_uti', 'proto_vines', 'proto_visa', 'proto_vmtp', 'proto_vrrp',
                       'proto_wb-expak', 'proto_wb-mon', 'proto_wsn', 'proto_xnet', 'proto_xns-idp', 'proto_xtp',
                       'proto_zero', 'state_ACC', 'state_CLO', 'state_CON', 'state_ECO', 'state_ECR', 'state_FIN',
                       'state_INT', 'state_MAS', 'state_PAR', 'state_REQ', 'state_RST', 'state_TST', 'state_TXD',
                       'state_URH', 'state_URN', 'state_no', 'service_-', 'service_dhcp', 'service_dns', 'service_ftp',
                       'service_ftp-data', 'service_http', 'service_irc', 'service_pop3', 'service_radius',
                       'service_smtp', 'service_snmp', 'service_ssh', 'service_ssl', 'is_sm_ips_ports_0',
                       'is_sm_ips_ports_1', 'is_ftp_login_0', 'is_ftp_login_1', 'is_ftp_login_2', 'is_ftp_login_4',
                       'attack_cat_Analysis', 'attack_cat_Backdoor', 'attack_cat_DoS', 'attack_cat_Exploits',
                       'attack_cat_Fuzzers', 'attack_cat_Generic', 'attack_cat_Reconnaissance', 'attack_cat_Shellcode',
                       'attack_cat_Worms', 'attack_cat_Normal', 'label']