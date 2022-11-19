import sys
import os
import argparse
from data_utils import *
from model import *
from training_utils import *
import torch
import torch.nn as nn
import numpy as np


def _parse_args():

    parser = argparse.ArgumentParser(description='trainer.py')

    parser.add_argument('--dataset_size', type=int, default=10000, help='size of the synthetic dataset')
    parser.add_argument('--max_delay', type=int, default=8, help='maximum delay for each example in training dataset ')
    parser.add_argument('--seq_length', type=int, default=20, help='sequence length for each example in training dataset ')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--hidden_size', type=int, default=128, help='hidden size of GRU layer')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--optim', type=str, default='RMSprop', help='optimizer to use default is RMSprop for RNN')
    args = parser.parse_args()

    return args

if __name__ == '__main__':

    args = _parse_args()

    #create the dataset for training
    dataset_size = args.dataset_size
    max_delay = args.max_delay
    seq_length = args.seq_length
    dsv = VariableDelayEchoDataset(max_delay = max_delay, seq_length= seq_length, size = dataset_size)

    #create dataloader with the dataset
    batch_size = args.batch_size
    train_dataloader = torch.utils.data.DataLoader(dsv, batch_size=batch_size)

    #initial model, critetion, optimizer and start the training function
    hidden_size = args.hidden_size
    lr = args.lr
    num_epochs = args.num_epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VariableDelayGRUMemory(hidden_size, max_delay)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = getattr(optim, args.optim)(model.parameters(), lr = lr)

    model, accuracy, duration = train(model, train_dataloader,\
                                    optimizer, critetion, num_epochs,\
                                    test_variable_delay_model)
