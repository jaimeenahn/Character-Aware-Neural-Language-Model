import torch
import torch.nn as nn
import torch.nn.functional as F


def highway(input):

    g = nn.Tanh()
    W1 = nn.Linear(input.shape[1], input.shape[1], bias=True)
    W2 = nn.Linear(input.shape[1], input.shape[1], bias=True)
    t = F.sigmoid(W1(input))

    output1 = torch.mul(t, g(W2(input)))
    output2 = torch.mul((1. - t), input)
    output = output1 + output2

    return output

class CANLM(nn.Module) :
    def __init__ (self, word_vocab, char_vocab, max_len, embed_dim, out_channels, kernels, hidden_size, batch_size):
        super(CANLM, self).__init__()

        self.word_vocab = word_vocab
        self.char_vocab = char_vocab
        self.max_len = max_len
        self.embed_dim = embed_dim
        self.out_channel = out_channels
        self.kernels = kernels
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.input_highway = []
        #Embedding
        self.Embed = nn.Embedding(len(char_vocab)+1, embed_dim, padding_idx=0)

        #Char-CNN
        self.cnns=[]
        #LSTM
        self.lstm = nn.LSTM(525, hidden_size, 2, batch_first=True, dropout=0.5)

    def forward(self, x, h, c, W):
        #Embedding
        x = self.Embed(x)
        self.cnns = []
        #CNN
        x = x.reshape(self.batch_size * 35, 1, 21, 15)
        for kernel in self.kernels :
            conv = nn.Conv2d(1, self.out_channel * kernel, kernel_size=(kernel, self.embed_dim))(x)
            conv = F.tanh(conv)
            conv = torch.squeeze(conv)
            result = torch.max(conv, 2)[0]
            self.cnns.append(result)
            output = torch.cat((self.cnns), 1)
        high_in = output

        #Highway
        out_high = highway(high_in)
        out_high = out_high.reshape(20, 35, -1)

        #LSTM
        input_lstm = out_high
        out_lstm, (h_next, c_next) = self.lstm(input_lstm, (h, c))
        out_lstm = out_lstm.permute(1,0,2)
        out_per = [data for data in out_lstm]
        logits = list()
        for output_ in out_per :
            logits.append(F.softmax(W(output_)))

        return logits, (h_next, c_next), W

