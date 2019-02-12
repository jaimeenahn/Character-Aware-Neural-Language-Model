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
h = (Variable(torch.zeros(2, 20, hidden_size), requires_grad=True) , Variable(torch.zeros(2, 20, hidden_size), requires_grad=True))



model = CANLM(word_vocab, char_vocab, max_len, embed_dim, out_channels, kernels, hidden_size, batch_size, False)

old_PPL = 100000
best_PPL = 100000


#--------------------validation-------------------#



criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=1, verbose=True)

for epoch in range(num_epochs) :
    loss_batch = []
    PPL_batch = []

    model.eval()
    avg_loss = 0.0
    num_examples = 0
    # For training
    i = 0

    for input, target in zip(val_data, val_labels) :

        i += 1
        loss = 0

        h = [state.detach() for state in h]

        output, h = model(input, h = h)

       #target = target.type(torch.FloatTensor)
        val_loss = criterion(output.view(-1, 10000).squeeze(), target.view(-1))

        val_loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 5, 2)

        optimizer.step()
        model.zero_grad()


        if i % 5 == 0:
            print('Loss: %.3f, Perplexity: %5.2f' % (val_loss.data, np.exp(val_loss.data)))


    PPL = torch.exp(val_loss.data)

    loss_batch.append(float(val_loss))
    PPL_batch.append(float(PPL))

    PPL = np.mean(PPL_batch)
    print("[epoch {}] valid PPL={}".format(epoch, PPL))
    print("valid loss={}".format(np.mean(loss_batch)))
    print("PPL decrease={}".format(float(old_PPL - PPL)))


    # Adjust the learning rate
    if float(old_PPL - PPL) <= 1.0:
        learning_rate /= 2
        print("halved lr:{}".format(learning_rate))
    old_PPL = PPL


#--------------------train-------------------#


    model.train()

    avg_loss = 0.0
    num_examples = 0
    # For training
    i = 0

    for input, target in zip(data, labels) :

        i += 1
        loss = 0

        h = [state.detach() for state in h]

        output, h = model(input, h = h)

        #target = target.type(torch.FloatTensor)
        loss =criterion (output.view(-1, 10000), target.view(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 5, 2)

        optimizer.step()

        avg_loss += loss.item()
        model.zero_grad()
        num_examples += output.size(0)


        if i % 100 == 0:
            print("[epoch {} step {}] train loss={}, Perplexity={}".format(epoch + 1,
                                                                        i + 1, float(loss.data),
                                                                        float(np.exp(loss.data))))

torch.save(model.state_dict(), "cache/model.pt")
print("Training finished.")



