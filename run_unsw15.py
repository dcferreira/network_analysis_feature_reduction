import os
import random

import numpy as np
import tensorflow as tf

from data import UNSW15Data, SemisupUNSW15
from parser import parser


def get_data(parsed_args):
    if parsed_args.datatype == 'semisup':
        unsup = None
        if os.path.exists(os.path.join(parsed_args.datapath, 'unsup.csv')):
            unsup = os.path.join(parsed_args.datapath, 'unsup.csv')
        return SemisupUNSW15(os.path.join(parsed_args.datapath, 'train.csv'),
                             os.path.join(parsed_args.datapath, 'test.csv'),
                             unsup,
                             )
    else:
        return UNSW15Data(parsed_args.datapath + os.sep + 'UNSW-NB15_all.csv',
                          parsed_args.datapath + os.sep + 'UNSW_NB15_training-set.csv',
                          parsed_args.datapath + os.sep + 'UNSW_NB15_testing-set.csv')


if __name__ == '__main__':
    random.seed(1337)
    np.random.seed(1337)
    tf.set_random_seed(1337)

    args = parser.parse_args()
    data = get_data(args)
    args.func(args, data)
