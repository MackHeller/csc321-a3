# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 13:28:13 2017

@author: Mack Heller
"""

"""
Minimal character-level Vanilla RNN model. Written by Andrej Karpathy (@karpathy)
BSD License
"""
import numpy as np



# hyperparameters
hidden_size = 250 # size of hidden layer of neurons
seq_length = 25 # number of steps to unroll the RNN for

a = pickle.load(open("char-rnn-snapshot.pkl"))
# model parameters
Wxh = a["Wxh"] 
Whh = a["Whh"]
Why = a["Why"]
bh = a["bh"]
by = a["by"]
mWxh, mWhh, mWhy = a["mWxh"], a["mWhh"], a["mWhy"]
mbh, mby = a["mbh"], a["mby"]
#meta data info
chars, data_size, vocab_size, char_to_ix, ix_to_char = a["chars"].tolist(), a["data_size"].tolist(), a["vocab_size"].tolist(), a["char_to_ix"].tolist(), a["ix_to_char"].tolist()
print 'data has %d characters, %d unique.' % (data_size, vocab_size)

def gethiddenUnit(inputs, hprev):
  """
  inputs, list of integers.
  hprev is Hx1 array of initial hidden state
  """
  x = np.zeros((vocab_size, 1))
  # forward pass
  for t in xrange(len(inputs)):
    x = np.zeros((vocab_size,1)) # encode in 1-of-k representation
    x[inputs[t]] = 1
    hprev = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hprev) + bh) # hidden state
  return hprev
                

def sampleTemperature(h, seed_ix, n, temperature = 1.0):
  """ 
  sample a sequence of integers from the model 
  h is memory state, seed_ix is seed letter for first time step
  temperature, increases or reduces varience of the output. 
  """
  x = np.zeros((vocab_size, 1))
  x[seed_ix] = 1
  ixes = []
  np.random.seed(8)
  for t in xrange(n):
    h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
    y = np.dot(Why, h) + by
    y = y/temperature #temp step
    p = np.exp(y) / np.sum(np.exp(y))
    ix = np.random.choice(range(vocab_size), p=p.ravel())
    x = np.zeros((vocab_size, 1))
    x[ix] = 1
    ixes.append(ix)
  return ixes

def setStarterText(startString,hprev):
    inputIndexes = [char_to_ix[char] for char in startString]
    hprev = gethiddenUnit(inputIndexes, hprev)

    #get first output
    np.random.seed(8)
    y = np.dot(Why, h) + by
    y = y/temperature #temp step
    p = np.exp(y) / np.sum(np.exp(y))
    index = np.random.choice(range(vocab_size), p=p.ravel())
    txt = startString + ix_to_char(index)
    sample_ix = sampleTemperature(hprev, index, 200,0.5)
    txt += ''.join(ix_to_char[ix] for ix in sample_ix)
    print '----\n %s \n----' % (txt, )
    
def __main__(startString):
  n = 0
  while n<10:
    if n == 0:
        hprev = np.zeros((hidden_size,1)) # init RNN memory
    setStarterText(startString,hprev)
    
    n += 1 # iteration counter 
