import json
import argparse
import os
import numpy as np
import torch

class Dictionary(object):
    def __init__(self, path=''):
        self.word2idx = dict()
        self.idx2word = list()
        if path != '':  # load an external dictionary
            words = json.loads(open(path, 'r').readline())
            for item in words:
                self.add_word(item)

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pooling', type=str, default='all', 
                    help='the pooling method (e.g., all, mean, or max)')
    parser.add_argument('--emsize', type=int, default=300,
                        help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=300,
                        help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=2,
                        help='number of layers in BiLSTM')
    parser.add_argument('--attention-unit', type=int, default=350,
                        help='number of attention unit')
    parser.add_argument('--attention-hops', type=int, default=1,
                        help='number of attention hops, for multi-hop attention model')
    parser.add_argument('--model', type=str, default='BiLSTM',
                        help='which rnn model to be used, BiLSTM or BiGRU?'),
    parser.add_argument('--result', type=str, default='',
                        help='result filename'),
    parser.add_argument('--encoder', type=str, default='SAE_SingleHop',
                        help='which encoder to be used, SAE_SingleHop or SAE_MultiHop?')
    
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--clip', type=float, default=0.5,
                        help='clip to prevent the too large grad in LSTM')
    parser.add_argument('--nfc', type=int, default=512,
                        help='hidden (fully connected) layer size for classifier MLP')
    parser.add_argument('--lr', type=float, default=.001,
                        help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=40,
                        help='upper epoch limit')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='report interval')
    parser.add_argument('--save', type=str, default='',
                        help='path to save the final model')
    parser.add_argument('--dictionary', type=str, default='',
                        help='path to save the dictionary, for faster corpus loading')
    parser.add_argument('--word-vector', type=str, default='',
                        help='path for pre-trained word vectors (e.g. GloVe), should be a PyTorch model.')
    parser.add_argument('--train-data', type=str, default='',
                        help='location of the training data, should be a json file')
    parser.add_argument('--val-data', type=str, default='',
                        help='location of the development data, should be a json file')
    parser.add_argument('--test-data', type=str, default='',
                        help='location of the test data, should be a json file')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='batch size for training')
    parser.add_argument('--class-number', type=int, default=2,
                        help='number of classes')
    parser.add_argument('--optimizer', type=str, default='Adam',
                        help='type of optimizer')
    parser.add_argument('--penalization-coeff', type=float, default=1, 
                        help='the penalization coefficient')
    return parser.parse_args()


def makedirs(name):
    """helper function for python 2 and 3 to call os.makedirs()
       avoiding an error if the directory to be created already exists"""

    import os, errno

    try:
        os.makedirs(name)
    except OSError as ex:
        if ex.errno == errno.EEXIST and os.path.isdir(name):
            # ignore existing directory
            pass
        else:
            # a different error happened
            raise


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    # adapted from https://github.com/Bjarten/early-stopping-pytorch
    def __init__(self, dir, suffix, patience=5, verbose=False):
        """
        Args:
            dir: the directory where you want to store the model checkpoint.
            suffix: the file name for the saved the model checkpoint.
            patience (int): How long to wait after last time validation loss improved.
                            Default: 5
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.dir = dir
        self.suffix = suffix
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score <= self.best_score:
            self.counter += 1
            print(f'Early stopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decreases.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.dir + self.suffix)
        self.val_loss_min = val_loss


def count_parameters(model):
    # count how many parameters are there in the model
    return sum(p.numel() for p in model.parameters() if p.requires_grad)