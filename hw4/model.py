import torch
import torch.nn as nn
import math
from torch.autograd import Variable

# TODO: Your implementation goes here
def log_softmax(input_vectors):
    input_max, _ = torch.max(input_vectors, dim=2, keepdim=True)
    logsumexp_term = torch.log(
            torch.sum(
                torch.exp(
                    input_vectors - input_max,
                    ),
                dim=2,
                keepdim=True
                )
            ) + input_max
    return input_vectors - logsumexp_term
    
class Dropout(nn.Module):
    def __init__(self, p=0.5):
        super(Dropout, self).__init__()
        self.p = p
    def forward(self, input_vectors):
        if self.training:
            mask = Variable(torch.bernoulli(self.p + torch.zeros(input_vectors.size()))).cuda()
            return input_vectors * mask
        else:
            return input_vectors * self.p


class GRU(nn.Module):
    def __init__(self, input_dim, rnn_dim, direction='l'):
        super(GRU, self).__init__()    
        self.rnn_dim = rnn_dim
        self.direction = direction

        self.W = nn.Parameter(
                torch.rand(
                    input_dim,
                    rnn_dim
                    )
                )
        
        self.U = nn.Parameter(
                torch.rand(
                    rnn_dim,
                    rnn_dim
                    )
                )
        
        self.b = nn.Parameter(
                torch.rand(
                    1,
                    rnn_dim
                    )
                )

        self.W_r = nn.Parameter(
                torch.rand(
                    input_dim,
                    rnn_dim
                    )
                )
        
        self.U_r = nn.Parameter(
                torch.rand(
                    rnn_dim,
                    rnn_dim
                    )
                )
        

        self.b_r = nn.Parameter(
                torch.rand(
                    1,
                    rnn_dim
                    )
                )
        
        self.W_z = nn.Parameter(
                torch.rand(
                    input_dim,
                    rnn_dim
                    )
                )
        
        self.U_z = nn.Parameter(
                torch.rand(
                    rnn_dim,
                    rnn_dim
                    )
                )
        
        self.b_z = nn.Parameter(
                torch.rand(
                    1,
                    rnn_dim
                    )
                )


        self.init_hidden = nn.Parameter(
                torch.rand(
                    1,
                    rnn_dim
                    )
                )

        self.reset_parameters()
    
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(1.0 * self.rnn_dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)


    def forward(self, input_vectors):
        max_len = input_vectors.size(0)
        batch_size = input_vectors.size(1)
        prev_hidden = self.init_hidden.expand(batch_size, self.rnn_dim)
        hidden_list = []
        
        for i in range(max_len):
            hidden_list.append(prev_hidden.unsqueeze(0))
            if self.direction == 'l':
                rnn_input = input_vectors[i, :]
            elif self.direction == 'r':
                rnn_input = input_vectors[max_len - i - 1, :]
            else:
                raise 
            
            z = torch.sigmoid(
                    torch.mm(rnn_input, self.W_z) 
                    + torch.mm(prev_hidden, self.U_z) 
                    + self.b_z)

            
            r = torch.sigmoid(
                    torch.mm(rnn_input, self.W_r) 
                    + torch.mm(prev_hidden, self.U_r) 
                    + self.b_r)
            
            h_bar = torch.tanh(
                    torch.mm(rnn_input, self.W)
                    + torch.mm(r * prev_hidden, self.U) 
                    + self.b
                    )
            
            new_hidden = (1 - r) * prev_hidden + z * h_bar
            
            prev_hidden = new_hidden
        
        if self.direction == 'r':
            hidden_list = hidden_list[::-1]
        
        seq_hidden = torch.cat(hidden_list, dim=0)
        
        return seq_hidden

class LSTM(nn.Module):
    def __init__(self, input_dim, rnn_dim, direction='l'):
        super(LSTM, self).__init__()    
        self.rnn_dim = rnn_dim
        self.direction = direction

        self.weight = nn.Parameter(
                torch.rand(
                    input_dim + rnn_dim,
                    rnn_dim * 4
                    )
                )

        self.bias = nn.Parameter(
                torch.rand(
                    1,
                    rnn_dim * 4
                    )
                )

        self.init_hidden = nn.Parameter(
                torch.rand(
                    1,
                    rnn_dim
                    )
                )

        self.init_cell = nn.Parameter(
                torch.rand(
                    1,
                    rnn_dim
                    )
                )
        self.reset_parameters()
    
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(1.0 * self.rnn_dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)


    def forward(self, input_vectors):
        
        max_len = input_vectors.size(0)
        batch_size = input_vectors.size(1)
        
        prev_hidden = self.init_hidden.expand(batch_size, self.rnn_dim)
        prev_cell = self.init_cell.expand(batch_size, self.rnn_dim)
        
        hidden_list = []
        
        for i in range(max_len):
            hidden_list.append(prev_hidden.unsqueeze(0))
            if self.direction == 'l':
                rnn_input = input_vectors[i, :]
            elif self.direction == 'r':
                rnn_input = input_vectors[max_len - i - 1, :]
            else:
                raise 

            rnn_input = torch.cat([rnn_input, prev_hidden], dim=1)
            new_gates = torch.mm(rnn_input, self.weight) + self.bias

            f_gate = torch.sigmoid(new_gates[:, :self.rnn_dim])
            i_gate = torch.sigmoid(new_gates[:, self.rnn_dim : 2 * self.rnn_dim])
            o_gate = torch.sigmoid(new_gates[:, 2 * self.rnn_dim: 3 * self.rnn_dim])
            c_gate = torch.tanh(new_gates[:, 3 * self.rnn_dim:])
            
            new_cell = f_gate * prev_cell + i_gate * c_gate
            new_hidden = o_gate * torch.tanh(new_cell)
            
            prev_hidden = new_hidden
            prev_cell = new_cell
        
        if self.direction == 'r':
            hidden_list = hidden_list[::-1]
        
        seq_hidden = torch.cat(hidden_list, dim=0)
        
        return seq_hidden
        

class RNN(nn.Module):
    def __init__(self, input_dim, rnn_dim, direction='l'):
        super(RNN, self).__init__()    
        self.rnn_dim = rnn_dim
        self.direction = direction

        self.weight = nn.Parameter(
                0.1 * torch.rand(
                    input_dim + rnn_dim,
                    rnn_dim
                    )
                )

        self.bias = nn.Parameter(
                0.1 * torch.rand(
                    1,
                    rnn_dim
                    )
                )
        self.init_hidden = nn.Parameter(
                0.1 * torch.rand(
                    1,
                    rnn_dim
                    )
                )
    
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(1.0 * self.rnn_dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
        
    def forward(self, input_vectors):
        max_len = input_vectors.size(0)
        batch_size = input_vectors.size(1)
        prev_hidden = self.init_hidden.expand(batch_size, self.rnn_dim)
        hidden_list = []
        
        for i in range(max_len):
            hidden_list.append(prev_hidden.unsqueeze(0))
            if self.direction == 'l':
                rnn_input = input_vectors[i, :]
            elif self.direction == 'r':
                rnn_input = input_vectors[max_len - i - 1, :]
            
            rnn_input = torch.cat([rnn_input, prev_hidden], dim=1)
            
            new_hidden = torch.mm(rnn_input, self.weight) + self.bias
            
            new_hidden = torch.tanh(new_hidden)
            
            prev_hidden = new_hidden
            
        #hidden_list.append(prev_hidden.unsqueeze(0))
        
        if self.direction == 'r':
            hidden_list = hidden_list[::-1]
        seq_hidden = torch.cat(hidden_list, dim=0)
        
        return seq_hidden


class RNNLM(nn.Module):
    def __init__(self, vocab_size, rnn_type='RNN'):
        super(RNNLM, self).__init__()
        self.rnn_dim = 16
        self.emb_dim = 32

        self.embed = nn.Parameter(
                torch.rand(
                    vocab_size,
                    self.emb_dim
                    )
                )

        self.rnn_out = nn.Parameter(
                torch.rand(
                    self.rnn_dim,
                    vocab_size
                    )
                )
        self.rnn_out_bias = nn.Parameter(
                torch.rand(
                    1,
                    vocab_size
                    )
                )

        self.rnn = eval(rnn_type)(self.emb_dim, self.rnn_dim)

    def forward(self, input_batch):
        # embeding
        input_embed = self.embed[input_batch.data, :]
        
        # rnn
        seq_hidden = self.rnn(input_embed)
        seq_hidden = seq_hidden[1:, :, :]
        # seq_hidden shape : [len * batch * dim_model]
        # seq_prob shape : [len * batch * voc_size]
        
        seq_log_prob = log_softmax(
                torch.matmul(
                    seq_hidden,
                    self.rnn_out
                    ) + self.rnn_out_bias
                )
         
        #print(torch.sum(torch.exp(seq_log_prob), dim=2))
        #print(seq_log_prob)
        return seq_log_prob





# TODO: Your implementation goes here
class BiRNNLM(nn.Module):
    def __init__(self, vocab_size, rnn_type='GRU'):
        super(BiRNNLM, self).__init__()
        
        self.rnn_dim = 32
        self.emb_dim = 64
        self.vocab_size = vocab_size
        self.embed = nn.Parameter(
                torch.rand(
                    vocab_size,
                    self.emb_dim
                    )
                )

        self.rnn_out = nn.Parameter(
                torch.rand(
                    2 * self.rnn_dim,
                    vocab_size
                    )
                )
        
        self.rnn_out_bias = nn.Parameter(
                torch.rand(
                    1,
                    vocab_size
                    )
                )
        self.rnn_l = eval(rnn_type)(self.emb_dim, self.rnn_dim, direction='l')
        self.rnn_r = eval(rnn_type)(self.emb_dim, self.rnn_dim, direction='r')
    
    def forward(self, input_batch):
        # input_batch : [sequence_length, batch_size]
        # embeding
        
        input_embed = self.embed[input_batch.data, :]
       
        seq_hidden_l = self.rnn_l(input_embed)
        
        seq_hidden_r = self.rnn_r(input_embed)

        seq_hidden = torch.cat([seq_hidden_l, seq_hidden_r], dim=2)
      
        #seq_prob = torch.exp(torch.matmul(seq_hidden, self.rnn_out) + self.rnn_out_bias)
        #seq_prob = seq_prob / torch.sum(seq_prob, dim=2, keepdim=True)
        #seq_log_prob = torch.log(seq_prob)
        seq_log_prob = log_softmax(
                    torch.matmul(seq_hidden, self.rnn_out) + self.rnn_out_bias
                )
        return seq_log_prob
