import torch


def idx_to_onehot(x, k=26+1):
  """ Converts the generated integers to one-hot vectors """
  ones = torch.sparse.torch.eye(k)
  shape = x.shape
  res = ones.index_select(0, x.view(-1).type(torch.int64))
  return res.view(*shape, res.shape[-1])


class VariableDelayEchoDataset(torch.utils.data.IterableDataset):

  def __init__(self, max_delay=8, seq_length=20, size=1000):
    self.max_delay = max_delay
    self.seq_length = seq_length
    self.size = size
  
  def __len__(self):
    return self.size

  def __iter__(self):
    for _ in range(self.size):
      seq = torch.tensor([random.choice(range(1, N + 1)) for i in range(self.seq_length)], dtype=torch.int64)
      delay = random.randint(0, self.max_delay)
      result = torch.cat((torch.zeros(delay), seq[:self.seq_length - delay])).type(torch.int64)
      yield idx_to_onehot(seq), idx_to_onehot(torch.tensor(delay), k = self.max_delay+1 ), idx_to_onehot(result)