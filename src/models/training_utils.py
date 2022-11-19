import time
import random
import string
from tqdm.notebook import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F


def train(model, vtrain_dataloader, optimizer,criterion, num_epoch, test_variable_delay_model):
    start_time = time.time()

    pbar = tqdm(range(num_epoch))
    for epoch in pbar:
        for data, delay, target in tqdm(vtrain_dataloader, position=0, leave=False):
            model.train()

            data = data.to(device)
            target = target.to(device)
            delay = delay.to(device).view(batch_size, 1, MAX_DELAY+1)

            optimizer.zero_grad()
            out = model(data, delay)

            loss = criterion(out.permute(0,2,1), target.permute(0,2,1))
            loss.backward()
            optimizer.step()

    test_accu = test_variable_delay_model(model)
    pbar.set_postfix({'test acc': test_accu})


    end_time = time.time()
    train_time = end_time - start_time 
    return model, test_accu, train_time