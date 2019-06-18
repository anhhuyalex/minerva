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
correlated_data = np.load("../ising81x81_temp1.npy")[:10000,:9,:9]
data = np.vstack((uncorrelated_data, correlated_data))
label = np.hstack((-np.ones(10000), np.ones(10000)))
X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.33, random_state=42)


class IsingDataset(Dataset):
    def __init__(self, data, label):
        self.X = data
        self.y = label
        
        
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    
    def __len__(self):
        return len(self.X)
   

isingdataset = IsingDataset(X_train[:200], y_train[:200])
print(isingdataset.y)
# raise ValueError

# Create training and test dataloaders
num_workers = 0
# how many samples per batch to load
batch_size = 1
# number of epochs to train the model
n_epochs = 100
# learning rate
lr = 0.01
# adjust learning rate?
adjust_learning_rate = False

# specify loss function
criterion = nn.MSELoss()

# build model
model = supervised_convnet.SupervisedConvNet(filter_size = 3, square_size = 9)

# specify optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


# prepare data loaders
train_loader = torch.utils.data.DataLoader(isingdataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

for epoch in range(1, n_epochs+1):
    # monitor training loss
    accuracy = 0.0

    # adjust learning rate
    if adjust_learning_rate == True:
        supervised_convnet.adjust_learning_rate(optimizer, epoch, lr)

    ###################
    # train the model #
    ###################
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.unsqueeze(1).type('torch.FloatTensor')
        target = target.type('torch.FloatTensor')
        optimizer.zero_grad()
        output = model(data)[-1].view(-1)
        loss = criterion(output, target) 
        # add regularization
        # for param in model.parameters():
        #     loss += ((param)**2).sum()/200
        loss.backward()
        optimizer.step()

        # update running training loss
        accuracy += (torch.sign(output) == target).item()
    
    # print avg training statistics 
    # train_loss = train_loss/len(train_loader)
    if epoch % 1 == 0:
        print('Epoch: {} \tTraining Loss: {}'.format(
            epoch, 
            accuracy
            ))
#         print("data", data[:10])
        # print("output", (output)[:10])
        # print("target", (target)[:10])
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print (name, param.data)

for batch_idx, (data, target) in enumerate(train_loader):
    data = data.unsqueeze(1).type('torch.FloatTensor')#[0].unsqueeze(1)
    # print("data", data)
    target = target.type('torch.FloatTensor')
    optimizer.zero_grad()
    output = [i.view(-1) for i in model(data)]
    print("data", data[:10])
    print("output", (output[:10]))
    print("target", target[:10])
    v = torch.tensor([[[[-1., -1., -1., -1., -1., -1., -1., -1., -1.],
          [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
          [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
          [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
          [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
          [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
          [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
          [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
          [-1., -1., -1., -1., -1., -1., -1., -1., -1.]]]])
    print("correlated model(v)", model(v))
    v = torch.tensor([[[[ 1.,  1.,  1.,  1.,  1.,  1., -1., -1., -1.],
          [ 1.,  1.,  1.,  1.,  1.,  1., -1., -1., -1.],
          [ 1.,  1.,  1.,  1.,  1.,  1., -1., -1., -1.],
          [ 1.,  1.,  1.,  1.,  1.,  1., -1., -1., -1.],
          [ 1.,  1.,  1.,  1.,  1.,  1., -1., -1., -1.],
          [ 1.,  1.,  1.,  1.,  1.,  1., -1., -1., -1.],
          [-1., -1., -1.,  1.,  1.,  1., -1., -1., -1.],
          [-1., -1., -1.,  1.,  1.,  1., -1., -1., -1.],
          [-1., -1., -1.,  1.,  1.,  1., -1., -1., -1.]]]])
    print("uncorrelated model(v)", model(v))
    # loss = criterion(output, target[0])
    # print("loss.data", loss.data)
    # loss.backward()

v = torch.tensor([[[[-1., -1., -1., -1., -1., -1., -1., -1., -1.],
          [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
          [-1., -1., -1., -1.,  1., -1., -1., -1., -1.],
          [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
          [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
          [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
          [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
          [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
          [-1., -1., -1., -1., -1., -1., -1., -1., -1.]]]])
print("negative", model(v))
print("positive", model(-v))
for name, param in model.named_parameters():
    if param.requires_grad:
        print (name, param.data)

torch.save(model.state_dict(), "9x9->3x3.pt")
    # optimizer.step()
