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
from torch.utils.data import Dataset, DataLoader, TensorDataset
import boto3
from PIL import Image
import argparse
import json
import os
from tqdm import tqdm

s3 = boto3.client('s3', verify=True)

def test(model, test_loader):
    '''
    This function tests the model on the test data and prints the accuracy.

    Parameters:
    model: The model to test
    test_loader: The data loader for the test data
    '''
    model.eval()
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
    print("-> Loading pretrained model...")
    model = models.resnet18(pretrained=True)

    # Freeze the model's parameters
    if freeze_layers:
        print("-> Freezing pre-trained model layers...")
        for param in model.parameters():
            param.requires_grad = False # to freeze the pre-trained model's parameters

    # Replace the model's classifier with a new one
    num_classes = 10
    print(f"-> Replacing classifier with {num_classes} classes...")
    model.fc = nn.Linear(model.fc.in_features, num_classes) # replace the classifier with a new one for 10 classes
    
    return model
    
def create_or_read_s3_manifest(bucket, prefix, manifest_file='manifest.json'):

    # Check if manifest exists
    try:
        print("-> Loading manifest file...")
        response = s3.get_object(Bucket=bucket, Key=f'{prefix}/{manifest_file}')
        manifest = json.loads(response['Body'].read())
        print("-> Loaded an existing manifest file...")

        return manifest
    except s3.exceptions.NoSuchKey:
        print("-> Failed loading manifest file...")
        print("-> Creating a new manifest file...")
        manifest = []

    # Extract s3 metadata
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    for item in response.get('Contents', []):
        key = item['Key']
        key = item['Key']
        key_split = key.split("/")
        training_type = key_split[1]
        label_numeric = key_split[2].split(".")[0]
        label_description = key_split[2].split(".")[1]
        image_s3_path = "s3://" + os.path.join(bucket, key)

        # Add to manifest
        manifest.append({
            "image_ebs_path": os.path.join("temp", key),
            "image_s3_path": image_s3_path,
            "training_type": training_type,
            "label_numeric": label_numeric,
            "label_description": label_description
        })

    # Write manifest to S3
    s3_key_destination = f'{prefix}/{manifest_file}'
    s3.put_object(Bucket=bucket, Key=s3_key_destination, Body=json.dumps(manifest))
    
    return manifest

def download_images_to_ebs(metadata):

    print("-> Downloading images to local storage (EBS)...")

    for image_meta in tqdm(metadata):
        image_s3_path = image_meta['image_s3_path']
        local_image_path = image_meta['image_ebs_path']
        bucket = image_s3_path.split("/")[2]
        image_s3_key = "/".join(image_s3_path.split("/")[3:])

        # Download the image from S3 if it doesn't already exist locally
        if not os.path.exists(local_image_path):
            
            # Create the local directory if it doesn't exist
            local_dir = os.path.dirname(local_image_path)
            os.makedirs(local_dir, exist_ok=True)

            response = s3.get_object(Bucket=bucket, Key=image_s3_key)
            with open(local_image_path, 'wb') as f:
                f.write(response['Body'].read())

    print("-> All images downloaded to local storage (EBS).")

class CustomDataset(Dataset):
    def __init__(self, metadata, transform=None, training_type='train'):
        self.metadata = [item for item in metadata if item['training_type'] == training_type]
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        image_meta = self.metadata[idx]
        image_s3_path = image_meta['image_s3_path']
        bucket = image_s3_path.split("/")[2]
        image_s3_key = "/".join(image_s3_path.split("/")[3:])
        label_numeric = int(image_meta['label_numeric'])

        # Download the image from local storage (EBS)
        local_image_path = image_meta['image_ebs_path']
        image = Image.open(local_image_path).convert('RGB')  # Convert to RGB
        
        if self.transform:
            image = self.transform(image)

        return image, label_numeric

def create_data_loaders(metadata, batch_size, shuffle=True, num_workers=4):
    '''
    Creates data loaders for training and testing.

    Parameters:
    data: The preprocessed data and labels
    batch_size: The size of each mini-batch
    shuffle: Whether to shuffle the data (default is True)
    num_workers: The number of subprocesses to use for data loading (default is 4)

    Returns:
    train_loader: DataLoader for the training data
    test_loader: DataLoader for the test data
    '''
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the image to 224x224
        transforms.ToTensor(),          # Convert the image to a PyTorch tensor
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize the image
    ])

    # Create separate datasets for training and testing
    print("-> Creating custom data loaders...")
    train_dataset = CustomDataset(metadata, transform=transform, training_type='train')
    test_dataset = CustomDataset(metadata, transform=transform, training_type='test')

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader

def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
    model=net(freeze_layers=True)
    
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion_options = {
        "nll_loss": F.nll_loss,
        "cross_entropy": F.cross_entropy
    }
    loss_criterion = loss_criterion_options[args.criterion]

    optimizer_options = {
        "Adadelta": optim.Adadelta(model.parameters(), lr=args.lr),
        "Adam": optim.Adam(model.parameters(), lr=args.lr),
        "SGD": optim.SGD(model.parameters(), lr=args.lr)
    }
    optimizer = optimizer_options[args.optimizer]
    
    # Get metadata
    bucket = 'udacity-deeplearning-project'
    prefix = 'data'
    metadata = create_or_read_s3_manifest(bucket, prefix)

    # Download images to local storage (EBS)
    download_images_to_ebs(metadata)

    # Load the data
    train_loader, test_loader = create_data_loaders(metadata, args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers)
    
    print("*"*150)
    print("-> Starting model training...")
    for epoch in range(1, args.epoch + 1):
        '''
        TODO: Call the train function to start training your model
        Remember that you will need to set up a way to get training data from S3
        '''
        train(model, train_loader, loss_criterion, optimizer, epoch=args.epoch)
    
        '''
        TODO: Test the model to see its accuracy
        '''
        test(model, test_loader)
    
    '''
    TODO: Save the trained model
    '''
    torch.save(model, args.path)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Train dog breed classifier')
    '''
    TODO: Specify all the hyperparameters you need to use to train your model.
    '''
    parser.add_argument('--batch_size', type=int, default=64, help='batch size for training')
    parser.add_argument('--test-batch-size', type=int, default=1000, help='batch size for testing')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--epoch', type=int, default=10, help='number of epochs')
    parser.add_argument('--optimizer', type=str, default='Adadelta', help='optimizer to use')
    parser.add_argument('--criterion', type=str, default='nll_loss', help='loss criterion to use')
    parser.add_argument('--shuffle', type=bool, default=True, help='shuffle the training data')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers for data loading')
    parser.add_argument('--path', type=str, default='model.pth', help='path to save the trained model')
    args=parser.parse_args()

    main(args)
