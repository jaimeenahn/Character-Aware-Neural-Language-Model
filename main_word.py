import pickle
import torch
import model as mm


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    tensor = torch.LongTensor(idxs)
    return torch.autograd.Variable(tensor)

with open('data.pickle', 'rb') as f :
    data = pickle.load(f)

train_input = data['train_word']
train_label = data['train_word']
valid_input = data['valid_word']
valid_label = data['valid_word']
word_vocab = data['word_vocab']
char_vocab = data['char_vocab']


EMBEDDING_DIM = HIDDEN_DIM = 300
batch_size = 20

model = mm.LSTM(EMBEDDING_DIM, HIDDEN_DIM,
                   len(word_vocab),batch_size)
loss_function = torch.nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(35):
    for sentence, tags in zip(train_input, train_label):
        model.zero_grad()

        model.hidden = model.init_hidden()

        sentence_in = prepare_sequence(sentence, word_vocab)
        targets = prepare_sequence(tags, word_vocab)

        tag_scores = model(sentence_in)

        loss = loss_function(tag_scores, targets)
        print(", epoch: %d, loss: %1.3f" % (epoch + 1, loss.data[0]))

        loss.backward()
        optimizer.step()
