from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.nn as nn
import os

class Classifier_CNN(nn.Module):
    # rnn encoder
    def __init__(self, config):
        super(Classifier_CNN, self).__init__()
        self.dictionary = config['dictionary']
        self.drop = nn.Dropout(config['dropout'])
        self.embed = nn.Embedding(config['ntoken'], config['ninp'])
#        self.init_weights()
        self.embed.weight.data[self.dictionary.word2idx['<pad>']] = 0
        if os.path.exists(config['word-vector']):
            print('Loading word vectors from', config['word-vector'])
            vectors = torch.load(config['word-vector'])
            assert vectors[2] >= config['ninp']
            vocab = vectors[0]
            vectors = vectors[1]
            loaded_cnt = 0
            for word in self.dictionary.word2idx:
                if word not in vocab:
                    continue
                real_id = self.dictionary.word2idx[word]
                loaded_id = vocab[word]
                self.embed.weight.data[real_id] = vectors[loaded_id][:config['ninp']]
                loaded_cnt += 1
            print('%d words from external word vectors loaded.' % loaded_cnt)
        

        self.conv = nn.Conv2d(
                in_channels = 1,
                out_channels=200,
                kernel_size= (3, config['ninp'])
                )
        self.dense1 = nn.Sequential(
            nn.Linear(200, 100),
            nn.ReLU()
            )
        self.dense2 = nn.Linear(100,100)
        self.out = nn.Linear(100,10)

    def conv_block(self, inp, conv_layer):
        inp = torch.transpose(inp, 0, 1).contiguous().unsqueeze(1)
        # input.size() = (batch_size, 1, num_seq, embedding_length)
        conv_out = conv_layer(inp)# conv_out.size() = (batch_size, out_channels, dim, 1)
        activation = nn.functional.relu(conv_out.squeeze(3))# activation.size() = (batch_size, out_channels, dim1)
        max_out = nn.functional.max_pool1d(activation, activation.size()[2]).squeeze(2)# maxpool_out.size() = (batch_size, out_channels)
        
        return max_out
    # note: init_range constraints the value of initial weights
    # can try different initialization methods, e.g., Xavier initialization
    def init_weights(self, init_range=0.1):
        self.embed.weight.data.uniform_(-init_range, init_range)

    # input -> hidden output -> pooling
    def forward(self, inp, hidden):
        emb = self.drop(self.embed(inp))
        # outp: [len, bsz, nhid*2]
        outp = self.drop(self.conv_block(emb, self.conv))
        outp = self.dense1(outp)
        outp = self.dense2(outp)
        outp = self.out(outp)
        return outp, emb

    def init_hidden(self, bsz):
        b=1


class rnn_encoder(nn.Module):
    # rnn encoder
    def __init__(self, config):
        super(rnn_encoder, self).__init__()
        self.drop = nn.Dropout(config['dropout'])
        self.embed = nn.Embedding(config['ntoken'], config['ninp'])
        self.model  = config['model']
        # instantiate the rnn models
        # input of nn.LSTM/nn.GRU: (input_size, hidden_size, num_layers, bias, dropout)
        if self.model == 'BiLSTM':
            self.rnn = nn.LSTM(config['ninp'], config['nhid'], config['nlayers'], dropout=config['dropout'],
                              bidirectional=True)
        if self.model == 'BiGRU':
            self.rnn = nn.GRU(config['ninp'], config['nhid'], config['nlayers'], dropout=config['dropout'],
                              bidirectional=True)
        if self.model == 'LSTM':
            self.rnn = nn.LSTM(config['ninp'], config['nhid'], config['nlayers'], dropout=config['dropout'],
                              bidirectional=False)
        if self.model == 'GRU':
            self.rnn = nn.GRU(config['ninp'], config['nhid'], config['nlayers'], dropout=config['dropout'],
                              bidirectional=False)
        if self.model =='RCNN':
            self.rnn = nn.LSTM(config['ninp'], config['nhid'], config['nlayers'], dropout=config['dropout'],
                              bidirectional=True)
        self.nlayers = config['nlayers']
        self.nhid = config['nhid']
        self.pooling = config['pooling']
        self.dictionary = config['dictionary']
#        self.init_weights()
        self.embed.weight.data[self.dictionary.word2idx['<pad>']] = 0
        if os.path.exists(config['word-vector']):
            print('Loading word vectors from', config['word-vector'])
            vectors = torch.load(config['word-vector'])
            assert vectors[2] >= config['ninp']
            vocab = vectors[0]
            vectors = vectors[1]
            loaded_cnt = 0
            for word in self.dictionary.word2idx:
                if word not in vocab:
                    continue
                real_id = self.dictionary.word2idx[word]
                loaded_id = vocab[word]
                self.embed.weight.data[real_id] = vectors[loaded_id][:config['ninp']]
                loaded_cnt += 1
            print('%d words from external word vectors loaded.' % loaded_cnt)

    # note: init_range constraints the value of initial weights
    # can try different initialization methods, e.g., Xavier initialization
    def init_weights(self, init_range=0.1):
        self.embed.weight.data.uniform_(-init_range, init_range)

    # input -> hidden output -> pooling
    def forward(self, inp, hidden):
        emb = self.drop(self.embed(inp))
        # outp: [len, bsz, nhid*2]
        outp = self.rnn(emb, hidden)[0]
        # different pooling methods:
        #   mean:   average across all hidden states
        #   max:    maximum across all hidden states
        #   all:    weighted average across all hidden states with attention mechanism
        if self.model == 'RCNN':
            outp = torch.cat((emb, outp), 2)
            outp = nn.functional.relu(outp)
            outp = outp.permute(1,2,0)
            maxpool = nn.MaxPool1d(outp.size()[2])
            outp = maxpool(outp).squeeze()
        elif self.pooling == 'mean':
            # out: [len, bsz, nhid*2]; pooling over time
            outp = torch.mean(outp, 0).squeeze()    
        elif self.pooling == 'max':
            # out: [len, bsz, nhid*2]; pooling over time
            outp = torch.max(outp, 0)[0].squeeze()  
        elif self.pooling == 'all' or self.pooling == 'all-word':
            # [len, bsz, nhid*2] -> [bsz, len, nhid*2]
            outp = torch.transpose(outp, 0, 1).contiguous()
        return outp, emb

    # initialize hidden units
    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.model == 'BiLSTM':
            return (Variable(weight.new(self.nlayers * 2, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers * 2, bsz, self.nhid).zero_()))
        if self.model == 'BiGRU':
            return Variable(weight.new(self.nlayers * 2, bsz, self.nhid).zero_())

class SSASE_encoder(nn.Module):
    # self-attentive encoder with multiple attention hops
    def __init__(self, config):
        super(SSASE_encoder, self).__init__()
        self.encoder = rnn_encoder(config)
        self.drop = nn.Dropout(config['dropout'])
        self.ws1 = nn.Linear(config['nhid'] * 2, config['attention-unit'], bias=False)
        self.ws2 = nn.Linear(config['attention-unit'], config['attention-hops'], bias=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)
        self.dictionary = config['dictionary']
#        self.init_weights()
        self.attention_hops = config['attention-hops']
        self.model = config['model']

    # initialize weights for the attention matrices
    # can try different initialization methods, e.g., Xavier initialization
    def init_weights(self, init_range=0.1):
        self.ws1.weight.data.uniform_(-init_range, init_range)
        self.ws2.weight.data.uniform_(-init_range, init_range)

    def forward(self, inp, hidden):
        outp = self.encoder.forward(inp, hidden)[0]
        size = outp.size()  # [bsz, len, nhid*2]
        compressed_embeddings = outp.view(-1, size[2])  # [bsz*len, nhid*2]
        transformed_inp = torch.transpose(inp, 0, 1).contiguous()  # [len, bsz] -> [bsz, len]
        transformed_inp = transformed_inp.view(size[0], 1, size[1])  # [bsz, len] -> [bsz, 1, len]
        concatenated_inp = [transformed_inp for i in range(self.attention_hops)]    # hop * [bsz, 1, len]
        concatenated_inp = torch.cat(concatenated_inp, 1)  # [bsz, hop, len]

        hbar = self.tanh(self.ws1(self.drop(compressed_embeddings)))  # [bsz*len, attention-unit]
        alphas = self.ws2(hbar).view(size[0], size[1], -1)  # [bsz, len, hop]
        alphas = torch.transpose(alphas, 1, 2).contiguous()  # [bsz, hop, len]
        penalized_alphas = alphas + (
            -10000 * (concatenated_inp == self.dictionary.word2idx['<pad>']).float())
            # [bsz, hop, len] + [bsz, hop, len]
        alphas = self.softmax(penalized_alphas.view(-1, size[1]))  # [bsz*hop, len]
        alphas = alphas.view(size[0], self.attention_hops, size[1])  # [bsz, hop, len]
        # outp: [bsz, hop, len].bmm([[bsz, len, nhid*2]]) -> [bsz, hop, nhid*2]
        # alphas: [bsz, hop, len]
        return torch.bmm(alphas, outp), alphas

    def init_hidden(self, bsz):
        return self.encoder.init_hidden(bsz)

class Classifier_RNN(nn.Module):
    # classifier with single attention head
    def __init__(self, config):
        super(Classifier_RNN, self).__init__()
        self.model = config['encoder']
        self.RCNN = config['model']
        # if we use simple pooling methods, e.g., mean/max pooling, then we just need rnn_encoder model
        if config['encoder'] == 'RNN':
            self.encoder = rnn_encoder(config)
            if config['model'] == 'BiGRU' or config['model'] == 'BiLSTM':
                self.fc = nn.Linear(config['nhid'] * 2, config['nfc'])
            else:
            	self.fc = nn.Linear(config['nhid'], config['nfc'])
        elif config['encoder'] == 'SSASE':
                self.encoder = SSASE_encoder(config)
                self.fc = nn.Linear(config['nhid'] * 2 * config['attention-hops'], config['nfc'])
        elif config['encoder'] == 'SAAAS':
                self.model = 'SAAAS'
                self.encoder = SSASE_encoder(config)
                self.fc = nn.Linear(config['nhid'] * 2, config['nfc'])
                self.perRow = attention_perRow(config)
        else:
            raise Exception('Error when initializing Classifier')
        self.drop = nn.Dropout(config['dropout'])
        self.tanh = nn.Tanh()
        if config['model'] =="RCNN":
            self.pred = nn.Linear(config['nhid'] * 2 + config['ninp'], config['class-number'])
        else:
            self.pred = nn.Linear(config['nfc'], config['class-number'])
        self.dictionary = config['dictionary']

#        self.init_weights()

    def init_weights(self, init_range=0.1):
        # todo: try different initialization tricks
        self.fc.weight.data.uniform_(-init_range, init_range)
        self.fc.bias.data.fill_(0)
        self.pred.weight.data.uniform_(-init_range, init_range)
        self.pred.bias.data.fill_(0)

    def forward(self, inp, hidden):
        outp, attention = self.encoder.forward(inp, hidden) # outp: [bsz, hop, nhid*2]; attention: [bsz, hop, len]
        if self.model == 'SAAAS':
            alpha = self.perRow.forward(outp) # outp: [bsz, hop, 1]
            #per word
            outp = torch.transpose(outp, 1, 2).contiguous() # [bsz, nhid*2, hop]
            fc_inp = torch.squeeze(outp.bmm(alpha)) # [bsz, nhid*2, hop] * [bsz, hop, 1] -> [bsz, nhid*2]
            fc = self.tanh(self.fc(self.drop(fc_inp)))    # [bsz, nhid*2] -> [bsz, nfc]
        elif self.RCNN == "RCNN":
            pred = self.pred(self.drop(outp))
            return pred, outp
        else:
            outp = outp.view(outp.size(0), -1)  # outp: [bsz, hop, nhid*2] -> [bsz, nhid*2*hop]
            fc = self.tanh(self.fc(self.drop(outp)))    # [bsz, nhid*2*hop] -> [bsz, nfc]
        pred = self.pred(self.drop(fc)) # [bsz, nfc] -> [bsz, nclass]

        if self.model == 'SAAAS' or self.model == 'SSASE':
            return pred, attention
        return pred, outp

    def init_hidden(self, bsz):
        return self.encoder.init_hidden(bsz)

    def encode(self, inp, hidden):
        return self.encoder.forward(inp, hidden)[0]

class attention_perRow(nn.Module):
    # classifier with label-wise attention network
    def __init__(self, config):
        super(attention_perRow, self).__init__()
        self.drop = nn.Dropout(config['dropout'])
        self.tanh = nn.Tanh()
        self.nhid = config['nhid'] * 2
        self.ws =nn.Linear(config['nhid'] * 2, 1, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inp): # inp: [bsz, hop, nhid * 2]
        size = inp.size()
        inp = inp.contiguous().view(-1, size[2]) # inp: [bsz*hop, nhid*2]
        
        alphas = self.tanh(self.ws(self.drop(inp))).view(size[0], size[1], -1)  # [bsz*hop, 1] -> [bsz, hop, 1]
        return alphas #[bsz, hop, 1]
