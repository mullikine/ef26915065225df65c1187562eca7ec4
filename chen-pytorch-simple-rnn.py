import torch
from torchsummary import summary
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch import optim
import numpy as np
import math, random

# Define the model

class SimpleRNN(nn.Module):
    def __init__(self, hidden_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size

        self.inp = nn.Linear(300, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size, 2, dropout=0.05)
        self.out = nn.Linear(hidden_size, 300)

    def step(self, input, hidden=None):
        # input = self.inp(input)[:,0]
        input = self.inp(input.view(1, -1)).unsqueeze(1)
        # input = self.inp(input)
        #  print(input.shape)
        output, hidden = self.rnn(input, hidden)
        output = self.out(output.squeeze(1))
        return output, hidden

    def forward(self, inputs, hidden=None, force=True, steps=0):
        if force or steps == 0: steps = len(inputs)
        outputs = Variable(torch.zeros(steps-1,300))
        i = 0
        while(True):
            if force or i == 0:
                input = inputs[i]
            else:
                input = output
            output, hidden = self.step(input, hidden)

            if(sum(output[0])==0 or sum(inputs[i]) ==0):
                # break
                return outputs, hidden
            outputs[i] = output
            i += 1


n_epochs = 100
n_iters = 50
hidden_size = 100

model = SimpleRNN(hidden_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


example = np.load("examplevec.npy")
exampletc = Variable(torch.from_numpy(example))
# # print(exampletc.detach().shape)
# print(example.shape)
# out = model(exampletc,None,True)
# print(out[0].shape)
# exit(0)

from collections import defaultdict

vectors = np.load('/var/smulliga/source/git/mullikine/codenames/dataset/glove.6B.300d.npy')
word_list = [w.lower().strip() for w in open('/home/shane/var/smulliga/source/git/mullikine/codenames/dataset/words')]
print('...Making word to index dict')

word_to_index = {w:i for i,w in enumerate(word_list)}

# print(type(vectors))
# print(np.sqrt(vectors**2).shape)

glove_vectors = np.sum(np.sqrt(vectors**2),1)
# print(glove_vectors.shape)

def vector_to_word(vec):
    vec_mag = np.sum(np.sqrt(vec**2))
    symarr = []
    for i in range(len(vectors)):
        sim = np.dot(vectors[i],vec) / vec_mag*glove_vectors[i]
        symarr.append(sim)
    return word_list[np.argmax(np.array(symarr))]

def mse_vector_to_word(vec):
    symarr = []
    for i in range(len(vectors)):
        sim = np.sum(np.sqrt(np.square(vectors[i]-vec)))
        symarr.append(sim)
    return word_list[np.argmin(np.array(symarr))]

def t_vec_to_word(vec):
    symarr = np.mean((vec-vectors)**2,axis=1)

    return word_list[np.argmin(np.array(symarr))]

def word_to_vec(word):

    return vectors[word_to_index[word]]

def word_to_vector(word):
    return vectors[word_to_index[word]]


# def t_vec_to_word(vec):
#     criterion = nn.MSELoss()
#     vectort = Variable(torch.from_numpy(vectors))
#     vect  = Variable(torch.from_numpy(np.broadcast(vec,vectors)))
#     symarr = criterion(vect,vectort)

#     return word_list[np.argmin(np.array(symarr))]



# toprint = ""
# for vec in example:
#     toprint += t_vec_to_word(vec) + " "
# print(toprint.strip(" "))
#     # print(mse_vector_to_word(vec))




# symdict = defaultdict()



symarr = []




# ice_cream = defaultdict(lambda: 'Vanilla')

# self.out = nn.Linear(hidden_size, 1)



# summary(model, (2,1,300))

import os

from shanepy import *

train_dir = r'/home/shane/dump/home/shane/notes2018/ws/deep-learning/chen/traindata'
for epoch in range(n_epochs):
    # tv(os.listdir(train_dir))

    for i,train in enumerate(os.listdir(train_dir)):
        text = np.load(os.path.join(train_dir,train))

        inputs = Variable(torch.from_numpy(text[:]).float(),requires_grad=True)
        targets = Variable(torch.from_numpy(text[1:]).float(),requires_grad=True)

        # Use teacher forcing 50% of the time
        force = random.random() < 0.5
        outputs, hidden = model(inputs, None, force)

        optimizer.zero_grad()
        loss = 0.000001* criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        # losses[epoch] += loss.data[0]
        print(i,loss.data[0])
    if epoch > 0:
        print(epoch, loss.data[0])

    # 'model': Classifier(),
    checkpoint = {
              'state_dict': model.state_dict(),
              'optimizer' : optimizer.state_dict()}

    torch.save(checkpoint, 'checkpoint.pth')

from ptpython.repl import embed
embed(globals(), locals())