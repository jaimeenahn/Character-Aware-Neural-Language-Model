import numpy as np
import pickle
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
"""
class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_size):
        super(Model, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers=num_layers, bias=True, batch_first=True, dropout=0.5 ,bidirectional=True)
        #input size 520

    def forward(self, input_tensor, batch_size, num_layers, hidden_size):
        hidden = Variable(torch.zeros(num_layers, batch_size, hidden_size))  # (num_layers * num_directions, batch, hidden_size)
        cell = Variable(torch.zeros(num_layers, batch_size, hidden_size))  # (num_layers * num_directions, batch, hidden_size)

        output, (hidden, cell) = self.rnn(input_tensor, (hidden, cell))

        return hidden, output
    """


class LSTMTagger(torch.nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, batch_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.word_embeddings = torch.nn.Embedding(vocab_size,
                                                  embedding_dim)

        # LSTM은 word embedding과 hidden 차원값을 input으로 받고,
        # hidden state를 output으로 내보낸다.
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim)

        # Hidden state space에서 tag space로 보내는 linear layer를 준비한다.
        self.hidden2tag = torch.nn.Linear(hidden_dim, vocab_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Hidden state는 자동적으로 만들어지지 않으므로 직접 기능을 만들겠다.
        # 3D tensor의 차원은 각각 (layer 개수, mini-batch 개수, hidden 차원)
        # 을 의미한다. 왜 이렇게 해야만 하는지 궁금하다면 Pytorch 문서를 참고 바란다.
        return (
            torch.autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim)),
            torch.autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim)),
        )

    def forward(self, ):
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(
            embeds.view(len(sentence), batch_size, -1),
            self.hidden
        )
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = torch.nn.functional.log_softmax(tag_space, dim=1)
        return tag_scores
def highway(input, input_size):

    g = nn.tanh()
    W1 = nn.Linear(input_size, input_size, bias=True)
    W2 = nn.Linear(input_size, input_size, bias=True)
    t = F.sigmoid(W1(input))

    output = torch.mul(t , g(W2(input))) + torch.mul((1.- t) , input)

    return output

def charLSTM(linear, input_size) :
    hidden_size = 300
    output_size = input_size
    batch_size = 20
    length = 4
    num_layers = 1
    rnn = nn.LSTM(input_size, hidden_size, num_layers=num_layers, bias=True, batch_first=True, dropout=0.5 ,bidirectional=True)

    input = Variable(torch.randn(batch_size,length,input_size)) # B,T,D
    hidden = Variable(torch.zeros(num_layers*2,batch_size,hidden_size)) # (num_layers * num_directions, batch, hidden_size)
    cell = Variable(torch.zeros(num_layers*2,batch_size,hidden_size)) # (num_layers * num_directions, batch, hidden_size)

    output, (hidden,cell) = rnn(input,(hidden,cell))

    print(output.size())
    print(hidden.size())
    print(cell.size())


    linear = nn.Linear(hidden_size*2,output_size)
    output = F.softmax(linear(output),1)
    output.size()


with open('data.pickle', 'rb') as f :
    data = pickle.load(f)

train_input = data['train_char']
train_label = data['train_char']
valid_input = torch.from_numpy(data['valid_char'])
valid_label = torch.from_numpy(data['valid_char'])
word_vocab = data['word_vocab']
char_vocab = data['char_vocab']
input_size = 0

model = LSTMTagger(64, 300, word_vocab.size(), 20)

loss_function = torch.nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)


for epoch in range(100):
    optimizer.zero_grad()
    loss = 0
    hidden_size = 300
    output_size = input_size
    batch_size = 20
    length = 4
    num_layers = 2

    for input, label in zip(train_input, train_label):
        hidden, output = model(input_tensor = input, batch_size=batch_size, num_layers=num_layers, hidden_size=hidden_size)
        val, idx = output.max(1)
        loss += loss_function(output, label)

    print(", epoch: %d, loss: %1.3f" % (epoch + 1, loss.data[0]))
    loss.backward()
    optimizer.step()




