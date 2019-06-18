import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import supervised_convnet
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import sys
from sklearn.model_selection import train_test_split


uncorrelated_data = np.load("../ising81x81_temp1_uncorrelated9x9.npy")
correlated_data = np.load("../ising81x81_temp1.npy")[:,:9,:9]
data = np.vstack((uncorrelated_data, correlated_data))
label = np.hstack((np.zeros(10000), np.ones(10000)))
X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.33, random_state=42)


class IsingDataset(Dataset):
    def __init__(self, data, label):
        self.X = data
        self.y = label
        
        
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    
    def __len__(self):
        return len(self.X)
   

isingdataset = IsingDataset(X_train[:100], y_train[:100])

model = supervised_convnet.SupervisedConvNet(filter_size = 3, square_size = 9)

# specify loss function
criterion = nn.BCELoss()

# specify loss function
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# Create training and test dataloaders
num_workers = 0
# how many samples per batch to load
batch_size = 5
# number of epochs to train the model
n_epochs = 500

# prepare data loaders
train_loader = torch.utils.data.DataLoader(isingdataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

for epoch in range(1, n_epochs+1):
    # monitor training loss
    train_loss = 0.0

    ###################
    # train the model #
    ###################
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.unsqueeze(1).type('torch.FloatTensor')
        target = target.type('torch.FloatTensor')
        optimizer.zero_grad()
        output = model(data)[4].view(-1)
        loss = criterion(output, target)
        # add regularization
        # for param in model.parameters():
        #     loss += (param.view(-1)).sum()**2  
        loss.backward()
        optimizer.step()

        # update running training loss
        train_loss += loss.item() * batch_size
    
    # print avg training statistics 
    train_loss = train_loss/len(train_loader)
    if epoch % 1 == 0:
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(
            epoch, 
            train_loss
            ))
#         print("data", data[:10])
        # print("output", (output)[:10])
        # print("target", (target)[:10])
        for name, param in model.named_parameters():
            if param.requires_grad:
                print (name, param.data)

for batch_idx, (data, target) in enumerate(train_loader):
    data = data.unsqueeze(1).type('torch.FloatTensor')#[0].unsqueeze(1)
    print("data", data)
    target = target.type('torch.FloatTensor')
    optimizer.zero_grad()
    output = model(data)[-1].view(-1)
    print("output", output)
    print("target", target)
    # loss = criterion(output, target[0])
    # print("loss.data", loss.data)
    # loss.backward()
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print (name, param.grad)
    # optimizer.step()
