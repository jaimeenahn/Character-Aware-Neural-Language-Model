import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import pickle
from data_util import word_to_idx, to_char, txt_to_word
from models import CANLM

dic = pickle.load(open('data/dic.pickle', 'rb'))
word_vocab = dic['word_vocab']
char_vocab = dic['char_vocab']

# Parameters
batch_size = 20
max_len = dic['max_len'] + 2
embed_dim = 15
kernels = [1, 2, 3, 4, 5, 6]
out_channels = 25
seq_len = 35
hidden_size = 300
learning_rate = 1.0
num_epochs = 35

# load train set
data, num_batches = txt_to_word('data/train.txt', batch_size, seq_len)

# train_label
label = word_to_idx(data, word_vocab)

labels = label.copy()
labels[1:] = labels[:-1]
labels[-1] = label[0]

# train_input_data
to_char(data, char_vocab, max_len)
data = np.array(data)



data = torch.from_numpy(data)
labels = torch.from_numpy(labels)

data = data.view(batch_size, -1, max_len)
labels = labels.view(batch_size, -1)

data = data.type(torch.LongTensor)
labels = labels.type(torch.LongTensor)

labels = labels.reshape((-1, 20, 35))
data = data.reshape((-1, 20, 35, 21))
# load validation set
val_data, _ = txt_to_word('data/valid.txt', batch_size, seq_len)

# val_label
val_label = word_to_idx(val_data, word_vocab)

val_labels = val_label.copy()
val_labels[1:] = val_labels[:-1]
val_labels[-1] = val_label[0]

# val_input_data
to_char(val_data, char_vocab, max_len)
val_data = np.array(val_data)

val_labels = torch.from_numpy(val_labels)
val_data = torch.from_numpy(val_data)

val_data = val_data.view(batch_size, -1, max_len)
val_labels = val_labels.view(batch_size, -1)

val_data = val_data.type(torch.LongTensor)
val_labels = val_labels.type(torch.LongTensor)

val_labels = val_labels.reshape((-1, 20, 35))
val_data = val_data.reshape((-1, 20, 35, 21))


#(num_layers, batch, hidden_size)
h = Variable(torch.zeros(2, 20, hidden_size))
c = Variable(torch.zeros(2, 20, hidden_size))



model = CANLM(word_vocab, char_vocab, max_len, embed_dim, out_channels, kernels, hidden_size, batch_size)


#--------------------validation-------------------#

model.eval()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=1, verbose=True)

for epoch in range(num_epochs) :
    avg_loss = 0.0
    num_examples = 0
    # For training
    model.train()
    i = 0

    for input, target in zip(val_data, val_labels) :

        i += 1
        loss = 0

        h = h.detach()
        c = c.detach()

        output, (h, c) = model(input, h = h, c = c)

        #target.from_numpy(np.asarray(target)).long()
        loss = nn.CrossEntropyLoss()(output.view(-1, 10000), target.view(-1))
        model.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 5, 2)

        optimizer.step()
        avg_loss += loss.item()

        num_examples += output.size(0)

        if i % 25 == 0:
            print('Loss: %.3f, Perplexity: %5.2f' % (loss.data, np.exp(loss.data)))

    avg_loss /= num_examples
    print('Epoch [%d/%d], Avg_Loss : %.3f ,Loss: %.3f, Perplexity: %5.2f' % (
        epoch + 1, num_epochs, avg_loss, loss, np.exp(loss.item())))





#--------------------train-------------------#

model.train()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=1, verbose=True)

for epoch in range(num_epochs) :
    avg_loss = 0.0
    num_examples = 0
    # For training
    model.train()
    i = 0

    for input, target in zip(data, labels) :

        i += 1
        loss = 0

        h = h.detach()
        c = c.detach()

        output, (h, c) = model(input, h = h, c = c)

        #target.from_numpy(np.asarray(target)).long()
        loss = nn.CrossEntropyLoss()(output.view(-1, 10000), target.view(-1))
        model.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 5, 2)

        optimizer.step()
        avg_loss += loss.item()

        num_examples += output.size(0)

        if i % 25 == 0:
            print('Loss: %.3f, Perplexity: %5.2f' % (loss.data, np.exp(loss.data)))

    avg_loss /= num_examples
    print('Epoch [%d/%d], Avg_Loss : %.3f ,Loss: %.3f, Perplexity: %5.2f' % (
        epoch + 1, num_epochs, avg_loss, loss, np.exp(loss.item())))

