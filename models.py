import torch
import torch.nn as nn
import torch.nn.functional as F


class highway(nn.Module):
    def __init__ (self, input_size):
        super(highway, self).__init__()
        self.W1 = nn.Linear(input_size, input_size, bias=True)
        self.W2 = nn.Linear(input_size, input_size, bias=True)

    def forward(self, input):
        g = nn.Tanh()
        t = F.sigmoid(self.W1(input))
        output1 = torch.mul(t, g(self.W2(input)))
        output2 = torch.mul((1. - t), input)
        output = output1 + output2
        return output

class CANLM(nn.Module) :
    def __init__ (self, word_vocab, char_vocab, max_len, embed_dim, out_channels, kernels, hidden_size, batch_size):
        super(CANLM, self).__init__()
        #Parameters
        self.word_vocab = word_vocab
        self.char_vocab = char_vocab
        self.max_len = max_len
        self.embed_dim = embed_dim
        self.out_channel = out_channels
        self.kernels = kernels
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        #Embedding
        self.Embed = nn.Embedding(len(char_vocab)+1, embed_dim, padding_idx=0)

        #Char-CNN
        self.cnns=[]
        for kernel in  self.kernels :
            self.cnns.append(nn.Conv2d(1, self.out_channel * kernel, kernel_size=(kernel, self.embed_dim)))

        #Highway
        self.highway = highway(sum([x for x in self.out_channel * self.kernels]))

        #LSTM
        self.lstm = nn.LSTM(525, hidden_size, 2, batch_first=True, dropout=0.5)

        self.W = nn.Linear(hidden_size, len(word_vocab), bias=True)

    def forward(self, x, h, c):
        #Embedding
        x = self.Embed(x)
        #CNN
        cnn_results = []
        x = x.reshape(self.batch_size * 35, 1, 21, 15)
        for cnn in self.cnns :
            conv = cnn(x)
            conv = F.tanh(conv)
            conv = torch.squeeze(conv)
            result = torch.max(conv, 2)[0]
            cnn_results.append(result)
            output = torch.cat((cnn_results), 1)
        high_in = output

        #Highway
        out_high = self.highway(high_in)
        out_high = out_high.reshape(20, 35, -1)

        #LSTM
        input_lstm = out_high
        out_lstm, (h_next, c_next) = self.lstm(input_lstm, (h, c))

        logits = F.softmax(self.W(out_lstm))

        return logits, (h_next, c_next)

