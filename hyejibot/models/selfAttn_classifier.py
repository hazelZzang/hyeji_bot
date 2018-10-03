import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable

class SelfAttention(nn.Module):
    def __init__(self, input_size, batch_size, hidden_dim, num_layers, dropout, n_classes):
        super().__init__()
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.cell = nn.LSTM(input_size, self.hidden_dim, num_layers=self.num_layers, dropout=dropout, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.self_attention = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim//2),
            nn.Tanh(),
            nn.Linear(self.hidden_dim//2, 1),
            nn.Softmax(dim=1)
        )
        self.L = nn.Linear(self.hidden_dim, n_classes)

    def init_hidden(self, batch_size):
        h0 = Variable(torch.zeros(2*self.num_layers, batch_size, self.hidden_dim))
        c0 = Variable(torch.zeros(2*self.num_layers, batch_size, self.hidden_dim))
        return h0, c0

    def forward(self, x):
        outputs, hidden = self.cell(x, self.init_hidden(x.size(0)))
        outputs = self.drop(outputs)
        outputs = outputs[:,:,:self.hidden_dim]+outputs[:,:,self.hidden_dim:] # B * C * H
        attention = self.self_attention(outputs)
        sentence_embeddings = attention.transpose(1,2)@outputs
        avg_sentence_embeddings = torch.sum(sentence_embeddings, 1)
        return F.log_softmax(self.L(avg_sentence_embeddings), dim=-1)

