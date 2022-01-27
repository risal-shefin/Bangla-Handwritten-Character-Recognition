import pandas as pd 
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import csv
import PIL

# custom dataset
class CustomDataset(Dataset):
    def __init__(self, images, labels=None, transforms=None):
        self.X = images
        self.y = labels
        self.transforms = transforms
         
    def __len__(self):
        return (len(self.X))
    
    def __getitem__(self, i):
        data = self.X.iloc[i, :]
        data = np.asarray(data).astype(np.float).reshape(28, 28, 1)
        
        if self.transforms:
            data = self.transforms(data)
        
        if self.y is not None:
            return (data, self.y[i])
        else:
            return data

def read_from_csv(train_file, test_file):
    # read the data
    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)
    
    # get the image pixel values and labels
    train_labels = df_train.iloc[:, 0]  # read the labels from the 0th column
    train_images = df_train.iloc[:, 1:] # read image pixels from the 1st column to the last column
    test_labels = df_test.iloc[:, 0]  # if test set has no label its none
    test_images = df_test.iloc[:, 1:]   # read image pixels from the 0th column to the last column
    del df_train; del df_test

    train_images = train_images.apply(lambda x: (x-x.min())/(x.max()-x.min()), axis=1)
    test_images = test_images.apply(lambda x: (x-x.min())/(x.max()-x.min()), axis=1)

    # make the labels starting from 0
    #train_labels = train_labels.apply(lambda x : x-1)
    #test_labels = test_labels.apply(lambda x : x-1)

    train_data = CustomDataset(train_images, train_labels, transforms.Compose([transforms.ToTensor()]))
    test_data = CustomDataset(test_images, test_labels, transforms.Compose([transforms.ToTensor()]))

    return train_data, test_data

def read_from_csv_train(train_file):
    # read the data
    df_train = pd.read_csv(train_file)
    
    # get the image pixel values and labels
    train_labels = df_train.iloc[:, 0]  # read the labels from the 0th column
    train_images = df_train.iloc[:, 1:] # read image pixels from the 1st column to the last column

    train_data = CustomDataset(train_images, train_labels, transforms.Compose([transforms.ToTensor()]))

    return train_data

def read_from_csv_aug_lu(train_file):
    # read the data
    df_train = pd.read_csv(train_file)
    
    # get the image pixel values and labels
    train_labels = df_train.iloc[:, 0]  # read the labels from the 0th column
    train_images = df_train.iloc[:, 1:] # read image pixels from the 1st column to the last column
    del df_train

    train_images = train_images.apply(lambda x: (x-x.min())/(x.max()-x.min()), axis=1)

    to_tranform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Pad([7, 7, 0, 0]),
                                     transforms.Resize((28,28))])

    train_data = CustomDataset(train_images, train_labels, to_tranform)

    return train_data

def read_from_csv_aug_lb(train_file):
    # read the data
    df_train = pd.read_csv(train_file)
    
    # get the image pixel values and labels
    train_labels = df_train.iloc[:, 0]  # read the labels from the 0th column
    train_images = df_train.iloc[:, 1:] # read image pixels from the 1st column to the last column
    del df_train

    train_images = train_images.apply(lambda x: (x-x.min())/(x.max()-x.min()), axis=1)

    to_tranform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Pad([7, 0, 0, 7]),
                                     transforms.Resize((28,28))])

    train_data = CustomDataset(train_images, train_labels, to_tranform)

    return train_data

def read_from_csv_aug_ru(train_file):
    # read the data
    df_train = pd.read_csv(train_file)
    
    # get the image pixel values and labels
    train_labels = df_train.iloc[:, 0]  # read the labels from the 0th column
    train_images = df_train.iloc[:, 1:] # read image pixels from the 1st column to the last column
    del df_train

    train_images = train_images.apply(lambda x: (x-x.min())/(x.max()-x.min()), axis=1)

    to_tranform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Pad([0, 7, 7, 0]),
                                     transforms.Resize((28,28))])

    train_data = CustomDataset(train_images, train_labels, to_tranform)

    return train_data

def read_from_csv_aug_rb(train_file):
    # read the data
    df_train = pd.read_csv(train_file)
    
    # get the image pixel values and labels
    train_labels = df_train.iloc[:, 0]  # read the labels from the 0th column
    train_images = df_train.iloc[:, 1:] # read image pixels from the 1st column to the last column
    del df_train
    
    train_images = train_images.apply(lambda x: (x-x.min())/(x.max()-x.min()), axis=1)

    to_tranform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Pad([0, 0, 7, 7]),
                                     transforms.Resize((28,28))])

    train_data = CustomDataset(train_images, train_labels, to_tranform)

    return train_data
