#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import argparse

def test(model, test_loader):
    '''
    This function tests the model on the test data and prints the accuracy.

    Parameters:
    model: The model to test
    test_loader: The data loader for the test data
    '''
    test_loss = 0
    correct = 0
    with torch.no_grad(): # this disables gradient computation which is not needed for testing
        for data, target in test_loader: # iterate over the test data
            output = model(data) # get the model's prediction
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item() # get the number of correct predictions

    test_loss /= len(test_loader.dataset) # calculate the average loss

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
        

def train(model, train_loader, criterion, optimizer, epoch):
    '''
    This function trains the model on the training data for each epoch.

    Parameters:
    model: The model to train
    train_loader: The data loader for the training data
    criterion: The loss criterion to use
    optimizer: The optimizer to use
    epoch: The current epoch number
    '''
    for batch_idx, (data, target) in enumerate(train_loader): # iterate over the training data
        optimizer.zero_grad() # zero the gradients for this batch to avoid accumulation of gradients from previous batches
        output = model(data) # get the model's prediction
        loss = criterion(output, target)  # use the criterion to calculate the loss
        loss.backward() # backpropagate the loss because we want to minimize it
        optimizer.step() # update the model's weights based on the gradients calculated during backpropagation
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    
def net(freeze_layers=True):
    '''
    This function initializes the model to be used for training. 

    Parameters:
    freeze_layers: A boolean that determines whether to freeze the pre-trained model's parameters or not
    '''
    # Load a pretrained model
    model = models.resnet18(pretrained=True)

    # Freeze the model's parameters
    if freeze_layers:
        for param in model.parameters():
            param.requires_grad = False # to freeze the pre-trained model's parameters

    # Replace the model's classifier with a new one
    num_classes = 10
    model.fc = nn.Linear(model.fc.in_features, num_classes) # replace the classifier with a new one for 10 classes

def create_data_loaders(data, batch_size, shuffle=True, num_workers=4):
    '''
    Creates data loaders for training and testing.

    Parameters:
    data: A tuple (train_data, train_labels, test_data, test_labels)
    batch_size: The size of each mini-batch
    shuffle: Whether to shuffle the data (default is True)
    num_workers: The number of subprocesses to use for data loading (default is 4)

    Returns:
    train_loader: DataLoader for the training data
    test_loader: DataLoader for the test data
    '''
    train_data, train_labels, test_data, test_labels = data

    # Create Tensor datasets
    train_dataset = TensorDataset(torch.tensor(train_data, dtype=torch.float32),
                                  torch.tensor(train_labels, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(test_data, dtype=torch.float32),
                                 torch.tensor(test_labels, dtype=torch.long))

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader

def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
    model=net()
    
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = None
    optimizer = None
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    model=train(model, train_loader, loss_criterion, optimizer)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    test(model, test_loader, criterion)
    
    '''
    TODO: Save the trained model
    '''
    torch.save(model, path)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify all the hyperparameters you need to use to train your model.
    '''
    
    args=parser.parse_args()
    
    main(args)
