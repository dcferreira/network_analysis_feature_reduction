import os
import random

import numpy as np
import tensorflow as tf

from data import UNSW15Data
from parser import parser


def get_data(parsed_args):
    return UNSW15Data(parsed_args.datapath + os.sep + 'UNSW-NB15_all.csv',
                          parsed_args.datapath + os.sep + 'UNSW_NB15_training-set.csv',
                          parsed_args.datapath + os.sep + 'UNSW_NB15_testing-set.csv')


parser.add_argument('datapath', type=str, help='Path for data.\n'
                                               'This should be a directory with the following files: '
                                               '"UNSW-NB15_all.csv", "UNSW_NB15_training-set.csv", and '
                                               '"UNSW_NB15_testing-set.csv".')



if __name__ == '__main__':
    random.seed(1337)
    np.random.seed(1337)
    tf.set_random_seed(1337)

    args = parser.parse_args()
    data = get_data(args)
    args.func(args, data)
