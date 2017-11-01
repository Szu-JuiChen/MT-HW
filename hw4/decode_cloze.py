import utils.tensor
import utils.rand

import argparse
import dill
import logging
import pdb

import torch
from torch import cuda
from torch.autograd import Variable
# from example_module import BiRNNLM
from model import BiRNNLM

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

BLK = "<blank>"
BLKend = "<blank>\n"
PAD= '<pad>'

def greedy_new(model, seq, vocab):
    seq_prob = model.forward(Variable(seq[:, None]))[:,0,:]
    new_seq = torch.LongTensor(seq.size())
    new_seq.copy_(seq)
    blank_idx = []
    for i in range(seq.size(0)):
        choose_index = 0
        if vocab.itos[seq[i]] == BLK or vocab.itos[seq[i]] == BLKend:
            blank_idx.append(i)

            rank, value = torch.topk(seq_prob[i], 5, dim=0)
            #print(value[choose_index].data[0] == vocab.stoi[PAD])

            while value[choose_index].data[0] == vocab.stoi[PAD]:
                choose_index += 1
            #_, pred_idx = torch.max(seq_prob[i], dim=0)
            pred_idx = value[choose_index].data[0] 
            new_seq[i] = pred_idx
    
    new_seq_prob = model.forward(Variable(new_seq[:, None]))[:,0,:]
    
    for i in range(seq.size(0)):
        choose_index = 0
        if i in blank_idx:

            rank, value = torch.topk(new_seq_prob[i], 5, dim=0)
            #print(value[choose_index].data[0] == vocab.stoi[PAD])

            while value[choose_index].data[0] in [vocab.stoi[PAD], vocab.stoi[BLK]]:
                choose_index += 1
            #_, pred_idx = torch.max(seq_prob[i], dim=0)
            pred_idx = value[choose_index].data[0] 
            new_seq[i] = pred_idx
    
    new_seq_prob = model.forward(Variable(new_seq[:, None]))[:,0,:]
    
    for i in range(seq.size(0)):
        choose_index = 0
        if i in blank_idx:

            rank, value = torch.topk(new_seq_prob[i], 5, dim=0)
            #print(value[choose_index].data[0] == vocab.stoi[PAD])

            while value[choose_index].data[0] in [vocab.stoi[PAD], vocab.stoi[BLK]]:
                choose_index += 1
            #_, pred_idx = torch.max(seq_prob[i], dim=0)
            pred_idx = value[choose_index].data[0] 
            new_seq[i] = pred_idx
    return new_seq

def greedy(model, seq, vocab):
    seq_prob = model.forward(Variable(seq[:, None]))[:,0,:]
    new_seq = torch.LongTensor(seq.size())
    new_seq.copy_(seq)
    
    for i in range(seq.size(0)):
        choose_index = 0
        if vocab.itos[seq[i]] == BLK or vocab.itos[seq[i]] == BLKend:
            rank, value = torch.topk(seq_prob[i], 5, dim=0)
            #print(value[choose_index].data[0] == vocab.stoi[PAD])

            while value[choose_index].data[0] == vocab.stoi[PAD]:
                choose_index += 1
            #_, pred_idx = torch.max(seq_prob[i], dim=0)
            pred_idx = value[choose_index].data[0] 
            new_seq[i] = pred_idx
    return new_seq


def tensor2string(tensor, vocab):
    string = []
    for i in range(tensor.size(0)):
        word = vocab.itos[tensor[i]]
        if word != '<s>' and word != '</s>':
            string.append(word)
    return u' '.join(string).encode('utf-8')


def main(options):
    train, dev, test, vocab = torch.load(open(options.data_file, 'rb'), pickle_module=dill)
    model = torch.load(options.model_file)
    #print(type(model)) 
    model.cpu()    
    strings = []
    new_seqs = []
    for seq in test:
        new_seq = greedy_new(model, seq, vocab)
        #new_string = tensor2string(new_seq, vocab)
        #strings.append(new_string)
        new_seqs.append(new_seq)
        print(tensor2string(seq, vocab))
        print(tensor2string(new_seq, vocab))
        '''
        print(seq)
        raw_input()
        print(new_seq)
        raw_input()
        '''

    with open(options.output_file, 'w+') as f_w, open(options.test_file, 'r') as f_r:
        seq_index = 0
        Ans=[]
        for test_seq in f_r:
            Ans.append([])
            test_seq_split = test_seq.split(' ')
            for index, word in enumerate(test_seq_split):
                if word == BLK or word == BLKend:
                   test_seq_split[index] = vocab.itos[new_seqs[seq_index][index + 1]] # due to there are " ' " in the begin and end
                   Ans[-1].append(vocab.itos[new_seqs[seq_index][index + 1]])
            new_seq = u' '.join(Ans[-1]).encode('utf-8')
            new_seq += '\n'
            f_w.write(new_seq)
            
            seq_index += 1
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decoding part of mt hw 4")
    parser.add_argument("--data_file", required=True,
            help="File for training set.")
    parser.add_argument("--model_file", required=True,
            help="Location to dump the models.")
    parser.add_argument("--output_file", required=True,
            help="Location for output.")
    parser.add_argument("--test_file", required=True,
            help="Location for test_file.")
    ret = parser.parse_known_args()
    
    options = ret[0]
    if ret[1]:
        logging.warning("unknown arguments: {0}".format(parser.parse_known_args()[1]))
    
    main(options)


