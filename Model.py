# .cuda() for using cuda enabled NVIDIA GPU to compute
# erase .cuda() if you haven't cuda enabled NVIDIA GPU

import torch

class Model(torch.nn.Module):
    def __init__(self):

        super(Model, self).__init__()

        # defining convolution layer 1
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1).cuda()
        
        # defining ReLU and max-pool layer
        self.relu = torch.nn.ReLU().cuda()
        self.max_pool2d = torch.nn.MaxPool2d(kernel_size=2, stride=2).cuda()
        self.max_pool2d_s = torch.nn.MaxPool2d(kernel_size=3, stride=2).cuda()
        self.max_pool2d_dec = torch.nn.MaxPool2d(kernel_size=2, stride=1).cuda()
        
        # defining convolution layer 2
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1).cuda()
        # defining second convolution layer 2 for more accuracy
        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1).cuda()

        # defining batch normalization
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.bn3 = torch.nn.BatchNorm2d(128)
        self.bn4 = torch.nn.BatchNorm1d(850)

        # defining the two fully connected layers
        self.linear1 = torch.nn.Linear(64*3*3, 850).cuda()
        self.linear2 = torch.nn.Linear(850, 232).cuda()

        # defining connection dropout probability to reduce overfitting
        self.dropout1 = torch.nn.Dropout(p=0.2).cuda()
        self.dropout2 = torch.nn.Dropout(p=0.4).cuda()

    # defining how the data will flow through the layers
    def forward(self, x):

        x = x.float().cuda()

        x = self.conv1(x).cuda()
        x = self.relu(x).cuda()
        x = self.max_pool2d(x).cuda()
        x = self.dropout1(x).cuda()

        x = self.conv2(x).cuda()
        x = self.relu(x).cuda()
        x = self.max_pool2d(x).cuda()
        x = self.dropout1(x).cuda()

        # Branching starts 

        y,z = x,x

        y = self.conv3(y).cuda()
        y = self.relu(y).cuda()
        y = self.max_pool2d_dec(y).cuda()
        y = self.dropout1(y).cuda()

        z = self.conv3(z).cuda()
        z = self.bn2(z).cuda()
        z = self.relu(z).cuda()
        z = self.max_pool2d_dec(z).cuda()

        x = y + z
        # Branching Ends

        x = self.conv3(x).cuda()
        x = self.relu(x).cuda()
        x = self.max_pool2d(x).cuda()
        x = self.dropout1(x).cuda()

        x = x.reshape(x.size(0), -1).cuda()

        x = self.linear1(x).cuda()
        x = self.bn4(x).cuda()

        x = self.relu(x).cuda()
        x = self.dropout2(x).cuda()

        ret = self.linear2(x).cuda()
        
        return ret.cpu()
