import random
import string
from tqdm.notebook import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F


class VariableDelayGRUMemory(torch.nn.Module):

  def __init__(self, hidden_size, max_delay):
    super().__init__()

    self.max_delay = max_delay
    self.hidden_size = hidden_size
    self.embed = nn.Linear(max_delay+1, 27)
    self.gru = nn.GRU(
        input_size = 27,
        hidden_size=hidden_size,
        num_layers=3,
        batch_first=True
    )
    self.linear1 = nn.Linear(hidden_size, 50)
    self.linear2 = nn.Linear(50, 27)

  def forward(self, x, delays):
    # inputs:
    # x - tensor of shape (batch size, seq length, N + 1)
    # delays - tensor of shape (batch size)
    # returns:
    # logits (scores for softmax) of shape (batch size, seq_length, N + 1)

    delays = self.embed(delays)
    x = x + delays
    outs, _= self.gru(x)
    outs = self.linear1(outs)
    outs = self.linear2(outs)
    return outs

  @torch.no_grad()
  def test_run(self, s, delay):
    # This function accepts one string s containing lowercase characters a-z, 
    # and a delay - the desired output delay.
    # we map those characters to one-hot encodings, 
    # then get the result from our network, and then convert the output 
    # back to a string of the same length, with 0 mapped to ' ', 
    # and 1-26 mapped to a-z.

    def s_to_i(s):
      i_list = []
      for i in s:
        if i == ' ':
          i_list.append(0)
        else:
          i_list.append(ord(i)-96)
      return i_list

    def i_to_s(pred):
      string = ''
      for i in pred:
        i = i.item()
        if i == 0:
          string += ' '
        else:
          string += (chr(i+96))
      return string

    i_list = s_to_i(s)
    i_list = idx_to_onehot(torch.tensor(i_list)).unsqueeze(0)
    delay = idx_to_onehot(torch.tensor(delay), k = self.max_delay+1).view(1,1,self.max_delay+1)
    out = self.forward(i_list.to(device), delay.to(device))
    pred = out.argmax(axis = -1).squeeze(0)
    out = i_to_s(pred)
    return out

def test_variable_delay_model(model, seq_length=20):
  """
  This is the test function that runs 100 different strings through our model,
  and checks the error rate.
  """
  total = 0
  correct = 0
  for i in range(500):
    s = ''.join([random.choice(string.ascii_lowercase) for i in range(seq_length)])
    d = random.randint(0, model.max_delay)
    result = model.test_run(s, d)
    if d > 0:
      z = zip(s[:-d], result[d:])
    else:
      z = zip(s, result)
    for c1, c2 in z:
      correct += int(c1 == c2)
    total += len(s) - d

  return correct / total