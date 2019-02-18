import os

from data import CSVData
from parser import parser


def get_paths(parsed_args):
    if parsed_args.datapath is not None or (parsed_args.train_path is not None and parsed_args.test_path is not None):
        raise ValueError('Either "datapath" is specified, or both "train_path" and "test_path" must be specified!')

    if parsed_args.datapath is None:
        train = parsed_args.train_path
        test = parsed_args.test_path
        unsup = parsed_args.unsup_path if parsed_args.unsup_path is not None else None
    else:
        train = os.path.join(parsed_args.datapath, 'train.csv')
        test = os.path.join(parsed_args.datapath, 'test.csv')
        unsup = None
        if os.path.exists(os.path.join(parsed_args.datapath, 'unsup.csv')):
            unsup = os.path.join(parsed_args.datapath, 'unsup.csv')
    return train, test, unsup


def get_data(train, test, unsup, fillna, read_args, label_name, cats_name):
    return CSVData(
        train,
        test,
        unsup,
        fillna=fillna,
        read_args=read_args,
        label_name=label_name,
        cats_name=cats_name,
    )


parser.add_argument('--datapath', type=str, default=None,
                    help='Path for data.\n'
                         'This should be a directory with the following files: '
                         '"train.csv", "test.csv", and (optionally) "unsup.csv".')
parser.add_argument('--train_path', type=str, default=None,
                    help='Path for training data.')
parser.add_argument('--test_path', type=str, default=None,
                    help='Path for test data.')
parser.add_argument('--unsup_path', type=str, default=None,
                    help='Path for unsupervised data.')
parser.add_argument('--fillna', type=float, default=0, help="Value with which to replace NaN in the CSV file.")
parser.add_argument('--read_args', type=str, default='None',
                    help='Arguments to pass to pandas when reading the csv file. '
                         'Note that this argument receives a string, which will be evaluated at runtime, '
                         'so the string needs to be valid Python code.')
parser.add_argument('--label_name', type=str, default='Label',
                    help='Name of the column containing the binary label.')
parser.add_argument('--cats_name', type=str, default='attack_cat',
                    help='Name of the column containing the categorical label.')


if __name__ == '__main__':
    args = parser.parse_args()

    train_path, test_path, unsup_path = get_paths(args)
    data = get_data(train_path, test_path, unsup_path,
                    args.fillna, eval(args.read_args), args.label_name, args.cats_name)
    args.func(args, data)
