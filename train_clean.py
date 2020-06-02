from __future__ import print_function
from model_clean import *

from util import Dictionary, get_args, makedirs, EarlyStopping

import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import json
import time
import random
import os

def Frobenius(mat):
    assert len( mat.shape )==3, 'matrix for computing Frobenius norm should be with 3 dims'
    return torch.sum( (torch.sum(torch.sum((mat ** 2), 2), 1) ) ** 0.5 )/mat.shape[0]


def package(data, volatile=False):
    """Package data for training / evaluation."""
    data = list(map(lambda x: json.loads(x), data))
    dat = list(map(lambda x: list(map(lambda y: dictionary.word2idx[y], x['text'])), data))
    maxlen = 0
    for item in dat:
        maxlen = max(maxlen, len(item))
    targets = list(map(lambda x: x['label'], data))
    maxlen = min(maxlen, 500)
    for i in range(len(data)):
        if maxlen < len(dat[i]):
            dat[i] = dat[i][:maxlen]
        else:
            for j in range(maxlen - len(dat[i])):
                dat[i].append(dictionary.word2idx['<pad>'])
    dat = Variable(torch.LongTensor(dat), volatile=volatile)
    targets = Variable(torch.FloatTensor(targets), volatile=volatile)
    return dat.t(), targets


def evaluate(eval_data='valid_data'):
    """evaluate the model while training"""
    model.eval()  # turn on the eval() switch to disable dropout
    with torch.no_grad():
        total_loss = 0
        true_positive = torch.zeros(args.class_number, device=device)
        false_positive = torch.zeros(args.class_number, device=device)
        true_negative = torch.zeros(args.class_number, device=device)
        false_negative = torch.zeros(args.class_number, device=device)

        rec_targets = torch.zeros(args.class_number, device=device)

        for batch, i in enumerate(range(0, len(data_val), args.batch_size)):
            if eval_data == 'valid_data':
                data, targets = package(data_val[i:min(len(data_val), i+args.batch_size)])
            elif eval_data == 'test_data':
                data, targets = package(data_test[i:min(len(data_val), i+args.batch_size)])       
            if args.cuda:
                data = data.cuda()
                targets = targets.cuda()
            hidden = model.init_hidden(data.size(1))
            output, attention = model.forward(data, hidden)
            output_flat = output.view(data.size(1), -1)
            total_loss += criterion(output_flat, targets).data

            rec_targets += targets.float().sum(dim=0)

            predictions = (output_flat.data > 0)
            true_positive += ((predictions == 1) & (targets == 1)).float().sum(dim=0)
            false_positive += ((predictions == 1) & (targets == 0)).float().sum(dim=0)
            true_negative += ((predictions == 0) & (targets == 0)).float().sum(dim=0)
            false_negative += ((predictions == 0) & (targets == 1)).float().sum(dim=0)

            # if eval_data == 'test_data':
            #     with open("_", "a") as f:
            #         # for i in (((predictions == 1) & (targets == 1)).int().cpu().numpy() + ((predictions == 0) & (targets == 0)).int().cpu().numpy()):
            #         for i in predictions:
            #             for j in i:
            #                 f.write(str(j)+'\t')
            #             f.write('\n')
            #     f.close()

        accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        F1 = 2 * (precision * recall) / (precision + recall)

        micro_accuracy = (true_positive + true_negative).sum() / (true_positive + true_negative + false_positive + false_negative).sum()
        micro_precision = (true_positive).sum() / (true_positive + false_positive).sum()
        micro_recall = true_positive.sum() / (true_positive + false_negative).sum()
        micro_F1 = 2*(micro_precision * micro_recall) / (micro_precision + micro_recall)

        eval_dict = {
            'accuracy': accuracy.cpu().numpy(),
            'precision': precision.cpu().numpy(),
            'recall': recall.cpu().numpy(),
            'F1': F1.cpu().numpy(),
            'micro_accuracy': micro_accuracy.cpu().numpy(),
            'micro_precision': micro_precision.cpu().numpy(),
            'micro_recall': micro_recall.cpu().numpy(),
            'micro_F1': micro_F1.cpu().numpy()
        }

        # if eval_data == 'valid_data':
        #     with open("parameter_tuning.txt", 'a') as f:
        #         f.write(str(eval_dict['micro_F1'])+"\n")
        #     f.close()
    return total_loss.item() / (len(data_val) // args.batch_size), eval_dict


def train(epoch_number):
    model.train()
    total_loss = 0
    total_pure_loss = 0  # without the penalization term
    start_time = time.time()
    for batch, i in enumerate(range(0, len(data_train), args.batch_size)):
        data, targets = package(data_train[i:i+args.batch_size], volatile=False)
        if args.cuda:
            data = data.cuda()
            targets = targets.cuda()
        hidden = model.init_hidden(data.size(1))
        output, attention = model.forward(data, hidden)
        loss = criterion(output.view(data.size(1), -1), targets)
        total_pure_loss += loss.data

        if args.encoder == 'SSASE' or args.encoder == 'SAAAS':
            if (attention.data != 0).sum() !=0:  # add penalization term
                # attention: [bsz, hop, len]; attentionT: [bsz, len, hop]
                attentionT = torch.transpose(attention, 1, 2).contiguous()  
                # [bsz, hop, len].bmm([bsz, len, hop]) -> [bsz, hop, hop]
                extra_loss = Frobenius(torch.bmm(attention, attentionT) - I[:attention.size(0)])    # to avoid overindexing
                loss += args.penalization_coeff * extra_loss
        
        optimizer.zero_grad()
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        total_loss += loss.data

        if batch % args.log_interval == 0 and batch > 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:3d}/{:3d} batches | s/batch {:3.2f} | loss {:5.3f} | pure loss {:5.3f}'.format(
                  epoch_number, batch, len(data_train) // args.batch_size,
                  elapsed / args.log_interval, total_loss / args.log_interval,
                  total_pure_loss / args.log_interval))
            total_loss = 0
            total_pure_loss = 0
            start_time = time.time()

    evaluate_start_time = time.time()
    val_loss, eval_dict = evaluate()
    print('-' * 140)
    fmt = '| evaluation | time: {:5.2f}s | valid loss {:4.3f} | Micro Accuracy {:4.3f} | Micro Precision {:4.3f} | Micro Recall {:4.3f} | Micro F1 {:4.3f} |'
    print(fmt.format((time.time() - evaluate_start_time), val_loss, eval_dict['micro_accuracy'], eval_dict['micro_precision'], eval_dict['micro_recall'], eval_dict['micro_F1']))
    print('-' * 140)

    # early stopping based on the micro F1 
    early_stopping(-eval_dict['micro_F1'], model)

if __name__ == '__main__':
	# parse the arguments
    args = get_args()

    # create the filename to save results/checkpoints
    if args.encoder == 'CNN':
        args.save = '_'.join([args.encoder, str(args.nfc), str(args.lr), str(args.dropout)])
        hyperpara = dict({
            'encoder': args.encoder,
            'lr': args.lr,
            'nfc': args.nfc,
            'dropout': args.dropout
        })
    if args.encoder == 'RNN':
        args.save = '_'.join([args.model, args.pooling, str(args.nhid), str(args.nfc), \
            str(args.lr), str(args.dropout)])

        hyperpara = dict({
            'model': args.model,
            'pooling': args.pooling,
            'nhid': args.nhid,
            'nfc': args.nfc,
            'lr': args.lr,
            'dropout': args.dropout
            })
    if args.encoder == 'SSASE':
        args.save = '_'.join([args.encoder, args.model, args.pooling, str(args.nhid), str(args.nfc), \
            str(args.attention_hops), str(args.attention_unit), str(args.lr), str(args.dropout)])

        hyperpara = dict({
            'encoder': args.encoder,
            'model': args.model,
            'pooling': args.pooling,
            'nhid': args.nhid,
            'nfc': args.nfc,
            'attention-hop': args.attention_hops,
            'attention-unit': args.attention_unit,
            'lr': args.lr,
            'dropout': args.dropout
        })
    if args.encoder == 'SAAAS':
        args.save = '_'.join([args.encoder, args.model, args.pooling, str(args.nhid), str(args.nfc), \
            str(args.attention_hops), str(args.attention_unit), str(args.lr), str(args.dropout)])

        hyperpara = dict({
            'encoder': args.encoder,
            'model': args.model,
            'pooling': args.pooling,
            'nhid': args.nhid,
            'nfc': args.nfc,
            'attention-hop': args.attention_hops,
            'attention-unit': args.attention_unit,
            'lr': args.lr,
            'dropout': args.dropout
        })

    save_dir = 'save_model'
    result_dir = save_dir + '/result/'
    ckpt_dir = save_dir + '/checkpoints/'
    makedirs(result_dir)
    makedirs(ckpt_dir)
    result_fn = result_dir + args.save + '.pkl'
    ckpt_fn = ckpt_dir + args.save + '.pt'

    # instantiate the early stopping object
    patience = 7
    early_stopping = EarlyStopping(ckpt_dir, args.save + '.pt', patience=patience, verbose=True)

    # Set the random seed manually for reproducibility.
    if args.cuda:
        if not torch.cuda.is_available():
            print("WARNING: No CUDA device detected, switch to cpu device!")
            device = torch.device('cpu')
            torch.manual_seed(args.seed)
        else:
            device = torch.device('cuda')
            torch.cuda.manual_seed(args.seed)
    else:
        if torch.cuda.is_available():
            print("WARNING: CUDA device detected, continue to use cpu device!")
            device = torch.device('cpu')
            torch.manual_seed(args.seed)
        else:
            device = torch.device('cpu')
            torch.manual_seed(args.seed)

    random.seed(args.seed)

    # Load Dictionary
    assert os.path.exists(args.train_data), "No training data detected!"
    assert os.path.exists(args.val_data), "No validation data detected!"
    assert os.path.exists(args.test_data), "No test data detected!"
    print('Begin to load the dictionary.')
    dictionary = Dictionary(path=args.dictionary)

    # n_token: number of tokens in the dictionary
    n_token = len(dictionary)

    # initialize the classifier; interesting way to use dictionary as input; more readable and better use it in the future
    # important: remember to change the type when switching to another model
    if args.encoder == "CNN":
        model = Classifier_CNN({
            'dropout': args.dropout,
            'ntoken': n_token,
            'ninp': args.emsize,
            'encoder': args.encoder,
            'nfc': args.nfc,
            'dictionary': dictionary,
            'word-vector': args.word_vector,
            'class-number': args.class_number,
        })
    elif args.encoder == "RNN":
        model = Classifier_RNN({
            'dropout': args.dropout,
            'ntoken': n_token,
            'nlayers': args.nlayers,
            'nhid': args.nhid,
            'ninp': args.emsize,
            'pooling': args.pooling,
            'encoder': args.encoder,
            'nfc': args.nfc,
            'dictionary': dictionary,
            'word-vector': args.word_vector,
            'class-number': args.class_number,
            'model': args.model
        })
    elif args.encoder == "SSASE" or args.encoder == "SAAAS":
        model = Classifier_RNN({
            'dropout': args.dropout,
            'ntoken': n_token,
            'nlayers': args.nlayers,
            'nhid': args.nhid,
            'ninp': args.emsize,
            'pooling': args.pooling,
            'encoder': args.encoder,
            'attention-unit': args.attention_unit,
            'attention-hops': args.attention_hops,
            'nfc': args.nfc,
            'dictionary': dictionary,
            'word-vector': args.word_vector,
            'class-number': args.class_number,
            'model': args.model
        })

    if args.cuda:
        model = model.cuda()

    # freeze word embeddings (seems to improve the results)
    if args.encoder == 'CNN':
        model.embed.weight.requires_grad = False
    elif args.encoder == "SSASE" or args.encoder == "SAAAS":
        model.encoder.encoder.embed.weight.requires_grad = False
    elif args.encoder == "RNN":
        model.encoder.embed.weight.requires_grad = False

    # print(args)
    I = Variable(torch.zeros(args.batch_size, args.attention_hops, args.attention_hops))
    for i in range(args.batch_size):
        for j in range(args.attention_hops):
            I.data[i][j][j] = 1
    if args.cuda:
        I = I.cuda()

    # loss function part 1: CrossEntropy loss
    criterion = nn.BCEWithLogitsLoss()
    # Which optimization method to use? Adam or SGD?
    if args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=[0.9, 0.999], eps=1e-8, weight_decay=0)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.01)
    elif args.optimizer == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr, alpha=0.99, eps=1e-08, momentum=0, weight_decay=0, centered=False)
    else:
        raise Exception('For other optimizers, please add it yourself. '
                        'supported ones are: SGD and Adam.')
    print('Begin to load data.')
    data_train = open(args.train_data).readlines()
    data_val = open(args.val_data).readlines()
    data_test = open(args.test_data).readlines()

    # data_train = data_train[:int(len(data_train)*0.8)]

    try:
        for epoch in range(args.epochs):
            train(epoch)
            if early_stopping.early_stop == True:
                print('Early stopping, early stopping counter exceeds the given patience at epoch {:d}!'.format(epoch+1))
                break;

        print('-' * 140)
        if early_stopping.early_stop == True or epoch == args.epochs:
            print('Finish training, start evaluation on test data!')
        print('-' * 140)
        
        # before evaluation, we need to load the golden model
        model.load_state_dict(torch.load(ckpt_fn))
        print(model)
        evaluate_start_time = time.time()
        test_loss, test_dict = evaluate('test_data')
        test_dict['hyperpara'] = hyperpara
        print('Evaluation result:')
        print('-' * 140)
        fmt = '| evaluation | time: {:5.2f}s | valid loss (pure) {:5.3f} | Micro Accuracy {:5.3f} | Micro Precision {:5.3f} | Micro Recall {:5.3f} | Micro F1 {:5.3f} |'
        print(fmt.format((time.time() - evaluate_start_time), test_loss, test_dict['micro_accuracy'], test_dict['micro_precision'], test_dict['micro_recall'], test_dict['micro_F1']))
        print('-' * 140)
        # with open(result_fn, 'wb') as f:
        #     pickle.dump(test_dict, f)
        # f.close()

        with open(args.result, 'a') as f:
            f.write("Precision: " + str(test_dict['micro_precision']) + " Recall: " + str(test_dict['micro_recall']) + " F1: " + str(test_dict['micro_F1']) + '\n')

    except KeyboardInterrupt:
        print('-' * 140)
        print('Exit from training early.')
        model.load_state_dict(torch.load(ckpt_fn))
        evaluate_start_time = time.time()
        test_loss, test_dict = evaluate('test_data')
        test_dict['hyperpara'] = hyperpara
        print('Evaluation result:')
        print('-' * 140)
        fmt = '| evaluation | time: {:5.2f}s | valid loss (pure) {:5.3f} | Micro Accuracy {:5.3f} | Micro Precision {:5.3f} | Micro Recall {:5.3f} | Micro F1 {:5.3f} |'
        print(fmt.format((time.time() - evaluate_start_time), test_loss, test_dict['micro_accuracy'], test_dict['micro_precision'], test_dict['micro_recall'], test_dict['micro_F1']))
        # with open(result_fn, 'wb') as f:
        #     pickle.dump(test_dict, f)
        # f.close()

        print('-' * 140)

        # with open(args.result, 'a') as f:
        #     f.write("Precision: " + str(test_dict['micro_precision']) + " Recall: " + str(test_dict['micro_recall']) + " F1: " + str(test_dict['micro_F1']) + '\n')

