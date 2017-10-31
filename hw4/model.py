import torch
import torch.nn as nn
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
    

class LSTM(nn.Module):
    def __init__(self, input_dim, rnn_dim):
        super(LSTM, self).__init__()    
        self.rnn_dim = rnn_dim

        self.weight = nn.Parameter(
                torch.FloatTensor(
                    input_dim + rnn_dim,
                    rnn_dim * 4
                    )
                )

        self.bias = nn.Parameter(
                torch.FloatTensor(
                    1,
                    rnn_dim * 4
                    )
                )

        self.init_hidden = nn.Parameter(
                torch.FloatTensor(
                    1,
                    rnn_dim
                    )
                )

        self.init_cell = nn.Parameter(
                torch.FloatTensor(
                    1,
                    rnn_dim
                    )
                )


    def forward(self, input_vectors):
        max_len = input_vectors.size(0)
        batch_size = input_vectors.size(1)
        prev_hidden = self.init_hidden.expand(batch_size, self.rnn_dim)
        prev_cell = self.init_cell.expand(batch_size, self.rnn_dim)
        hidden_list = []
        
        for i in range(max_len):
            hidden_list.append(prev_hidden.unsqueeze(0))
            rnn_input = torch.cat([input_vectors[i, :], prev_hidden], dim=1)
            new_gates = torch.mm(
                    rnn_input,
                    self.weight
                    ) + self.bias

            f_gate = torch.sigmoid(new_gates[:, :self.rnn_dim])
            i_gate = torch.sigmoid(new_gates[:, self.rnn_dim : 2 * self.rnn_dim])
            o_gate = torch.sigmoid(new_gates[:, 2 * self.rnn_dim: 3 * self.rnn_dim])
            input_v = torch.tanh(new_gates[:, 3 * self.rnn_dim:])
            
            new_cell = f_gate * prev_cell + i_gate * input_v
            new_hidden = o_gate * torch.tanh(new_cell)
            
            prev_hidden = new_hidden
        
        hidden_list.append(prev_hidden.unsqueeze(0))
        seq_hidden = torch.cat(hidden_list, dim=0)
        
        return seq_hidden
        

class RNN(nn.Module):
    def __init__(self, input_dim, rnn_dim):
        super(RNN, self).__init__()    
        self.rnn_dim = rnn_dim

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
    
        
    def forward(self, input_vectors):
        max_len = input_vectors.size(0)
        batch_size = input_vectors.size(1)
        prev_hidden = self.init_hidden.expand(batch_size, self.rnn_dim)
        hidden_list = []
        
        for i in range(max_len):
            hidden_list.append(prev_hidden.unsqueeze(0))
            rnn_input = torch.cat([input_vectors[i, :], prev_hidden], dim=1)
            new_hidden = torch.mm(
                    rnn_input,
                    self.weight
                    ) + self.bias
            new_hidden = torch.tanh(new_hidden)
            prev_hidden = new_hidden
            
        hidden_list.append(prev_hidden.unsqueeze(0))
         
        seq_hidden = torch.cat(hidden_list, dim=0)
        
        return seq_hidden


class RNNLM(nn.Module):
    def __init__(self, vocab_size):
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

        self.rnn = RNN(self.emb_dim, self.rnn_dim)

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
                    ) 
                )
         
        #print(torch.sum(torch.exp(seq_log_prob), dim=2))
        #print(seq_log_prob)
        return seq_log_prob





# TODO: Your implementation goes here
class BiRNNLM(nn.Module):
    def __init__(self, vocab_size, rnn_type='LSTM'):
        super(BiRNNLM, self).__init__()
        
        self.rnn_dim = 16
        self.emb_dim = 32

        self.embed = nn.Parameter(
                torch.FloatTensor(
                    vocab_size,
                    self.emb_dim
                    )
                )

        self.rnn_out = nn.Parameter(
                torch.FloatTensor(
                    2 * self.rnn_dim,
                    vocab_size
                    )
                )
        
        self.rnn_l = eval(rnn_type)(self.emb_dim, self.rnn_dim)
        self.rnn_r = eval(rnn_type)(self.emb_dim, self.rnn_dim)

    def forward(self, input_batch):
        # input_batch : [sequence_length, batch_size]
        # embeding
        input_embed = self.embed[input_batch.data, :]

        seq_hidden_l = self.rnn_l(input_embed)
        seq_hidden_r = self.rnn_r(input_embed)

        seq_hidden_l = seq_hidden_l[:-1] 
        seq_hidden_r = seq_hidden_r[:-1]
        seq_hidden = torch.cat([seq_hidden_l, seq_hidden_r], dim=2)

        seq_prob = log_softmax(
                torch.matmul(
                    seq_hidden,
                    self.rnn_out
                    )
                )
        return seq_prob
