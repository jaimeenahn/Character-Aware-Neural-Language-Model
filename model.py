import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable

class LSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, batch_size):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bias=True, batch_first=True, dropout=0.5)

        self.hidden2tag = nn.Linear(hidden_dim, vocab_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (
            Variable(torch.zeros(1, self.batch_size, self.hidden_dim)),
            Variable(torch.zeros(1, self.batch_size, self.hidden_dim)),
        )

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(embeds.view(len(sentence), 1, -1),self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


class LSTM_char(torch.nn.Module):
    def __init__(self, char_embedding_dim, char_hidden_dim, char_size, batch_size):
        super(LSTM_char, self).__init__()
        self.char_hidden_dim = char_hidden_dim
        self.hidden_dim = char_hidden_dim
        self.batch_size = batch_size
        self.char_embeddings = torch.nn.Embedding(char_size, char_embedding_dim)

        self.char_lstm = torch.nn.LSTM(char_embedding_dim, char_hidden_dim, bias=True, batch_first=True, dropout=0.5)


        self.hidden2tag = torch.nn.Linear(char_hidden_dim, char_size)
        self.char_hidden = self.init_char_hidden()

    def init_char_hidden(self):
        return (
            torch.autograd.Variable(torch.zeros(1, self.batch_size, self.char_hidden_dim)),
            torch.autograd.Variable(torch.zeros(1, self.batch_size, self.char_hidden_dim)),
        )

    def forward(self, sentence, sentence_char):
        for word in sentence_char:
            char_embeds = self.char_embeddings(sentence_char)
            char_lstm_out, self.char_hidden = self.char_lstm(char_embeds.view(len(sentence_char), 1, -1), self.char_hidden)

        tag_space = self.hidden2tag(char_lstm_out.view(len(sentence), -1))
        tag_scores = torch.nn.functional.log_softmax(tag_space, dim=1)
        return tag_scores