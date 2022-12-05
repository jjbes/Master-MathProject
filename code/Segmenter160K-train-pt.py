# -*- coding: utf-8 -*-
import time
import copy
import torch
from torch import nn
import torchvision
from torchvision import transforms
from sklearn.metrics import classification_report
import numpy as np
import torch.utils.data
from collections import Counter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

###########################################################
class NetCNN(nn.Module):
    def __init__(self):
        super(NetCNN, self).__init__()
        self.encode = nn.Sequential( 
            nn.Conv2d(1, 128, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 5)
        )
        self.fc = nn.Sequential(
            nn.Linear(2048 ,500),
            nn.ReLU(),
            nn.Linear(500,NB_CLASSES)
        )
    def forward(self, x):
        x = self.encode(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

###########################################################

class MetaClassDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, classes):
        self._dataset = dataset
        self._classes = classes
        self.classes = ["incorrect", "correct"]
        self.targets = [1 if t != self._classes.index("junk") else 0 for t in self._dataset.targets]
        self.imgs = list(map(lambda i:  (i[0], 1) if self._classes[i[1]] != "junk" else (i[0], 0), self._dataset.imgs))
        
    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        if self._classes[self._dataset[idx][1]] == "junk":
          _dataset = (self._dataset[idx][0], 0)
        else :
          _dataset = (self._dataset[idx][0], 1)
        return (_dataset)

# function to train the network
def train_cnn(net,  lr=1-5, epoch=10, verbosity=True):
  #Used for returned values
  val_err_array = np.array([])
  train_err_array = np.array([])
  nb_sample_array = np.array([])
  best_val_loss = 1000000
  best_nb_sample = 0
  best_model =  copy.deepcopy(net)
  loss_parameters={}

  #Validation display
  print_every = 200
  nb_used_sample = 0

  #Early stopping
  patience = 3
  trigger_times = 0

  #Criterion and Optimizer
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
  print(net)
  running_loss = 0.0
  num_epochs = epoch

  for epoch in range(num_epochs):
      start_time = time.time()
      for i, data in enumerate(train_dataloader, 0):

          images, labels = data   
          images, labels = images.to(device), labels.to(device)

          optimizer.zero_grad()
          outputs = net(images)
          loss = criterion(outputs, labels)       
          loss.backward()
          optimizer.step()
          running_loss += loss.item()
          nb_used_sample += MIN_BATCH_SIZE   

          if nb_used_sample % (print_every * MIN_BATCH_SIZE) == 0:
              train_err = (running_loss / (print_every * MIN_BATCH_SIZE))

              if(verbosity):
                print('Epoch %d batch %5d ' % (epoch + 1, i + 1))
                print('Train loss : %.3f' % train_err)

              running_loss = 0.0
              totalValLoss = 0.0
              with torch.no_grad():
                  for data in valid_dataloader:
                      images, labels = data
                      images, labels = images.to(device), labels.to(device)

                      outputs = net(images)
                      loss = criterion(outputs, labels)
                      totalValLoss += loss.item()
              val_err = (totalValLoss / len(valid_dataset))

              if(verbosity):
                print('Validation loss mean : %.3f' % val_err)

              if val_err <= best_val_loss:
                  trigger_times = 0
                  best_val_loss = val_err
                  best_nb_sample = nb_used_sample
                  best_model = copy.deepcopy(net)

              #Early stopping
              else:
                trigger_times += 1

                if trigger_times >= patience:
                    print("", end="\r")
                    print(f"Early stopping, best loss: {best_val_loss}")
                    return best_model, loss_parameters

              train_err_array = np.append(train_err_array, train_err)
              val_err_array = np.append(val_err_array, val_err)
              nb_sample_array = np.append(nb_sample_array, nb_used_sample)

              #Store loss parameters to display plot later
              loss_parameters = {"nb_sample_array":nb_sample_array, 
                                 "val_err_array":val_err_array, 
                                 "train_err_array":train_err_array, 
                                 "best_nb_sample":best_nb_sample, 
                                 "best_val_loss":best_val_loss}

          current_percentage = int(((i/len(train_dataloader))*(1/num_epochs)+(epoch/num_epochs))*100)
          if(not verbosity):
            print(f"{current_percentage}%", end=f'\r{current_percentage}%')
      if(verbosity):
        print("Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - start_time))

  print("", end="\r")
  print(f"Best loss: {best_val_loss}")

  return best_model, loss_parameters

# function to plot reporting on predicted data
def class_report(model, testloader, targets=None):
  y_pred = torch.Tensor().to(device)
  y_true = torch.Tensor().to(device)

  with torch.no_grad():
      for data in testloader:
          images, labels = data
          images, labels = images.to(device), labels.to(device)
          outputs = model(images)
          _, predicted = torch.max(outputs.data, 1)
          y_pred = torch.cat((y_pred, predicted))
          y_true = torch.cat((y_true, labels))
  print(classification_report(y_true.cpu(), y_pred.cpu(), target_names=targets))

###########################################################

MIN_BATCH_SIZE = 32

transform = transforms.Compose(
    [transforms.Grayscale(),
     transforms.ToTensor()])

fulltrainset = torchvision.datasets.ImageFolder(
    root='../data/CROHME+LargeJunk', transform=transform)

binary_trainset = MetaClassDataset(fulltrainset, fulltrainset.classes)
print(binary_trainset.classes)
print(Counter(binary_trainset.targets))

classes = binary_trainset.classes
NB_CLASSES = len(binary_trainset.classes)

a_part = int(len(binary_trainset) / 5)

train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
    binary_trainset, [3 * a_part, a_part, len(binary_trainset) - 4 * a_part])

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=MIN_BATCH_SIZE,
                                               shuffle=True, num_workers=0)

valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=MIN_BATCH_SIZE,
                                               shuffle=False, num_workers=0)

test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=MIN_BATCH_SIZE,
                                              shuffle=True, drop_last=True, num_workers=0)

print(len(train_dataloader)*32)
print(len(valid_dataloader)*32)
print(len(test_dataloader)*32)

###########################################################

net160k = NetCNN()
net160k.to(device)

segmenter160k, loss = train_cnn(net160k, lr=1e-3, epoch=10)

model_scripted = torch.jit.script(segmenter160k)
model_scripted.save('../model/segmenter_160K.pt')
