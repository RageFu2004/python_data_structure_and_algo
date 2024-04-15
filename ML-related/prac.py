import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from PIL import Image
dataset = torchvision.datasets.CIFAR10("../data",
                                       train=False,
                                       transform=torchvision.transforms.ToTensor(),
                                       download=True)
testdata = torchvision.datasets.CIFAR10("../data",
                                       train=True,
                                       transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset=dataset, batch_size=64)
testloader = DataLoader(dataset = testdata, batch_size=64)
total_test_len = len(testloader)
print(total_test_len)
class cnn(nn.Module):
    def __init__(self):
        super(cnn, self).__init__()
        self.conv1 = nn.Conv2d(3,32,5,padding=2)
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32,32,5,padding=2)
        self.maxpool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(32,64,5, padding=2)
        self.maxpool3 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(1024,64)
        self.linear2 = nn.Linear(64,10)

    def forward(self,x):
        x=self.conv1(x)
        x=self.maxpool1(x)
        x=self.conv2(x)
        x=self.maxpool2(x)
        x=self.conv3(x)
        x=self.maxpool3(x)
        x=self.flatten(x)
        x=self.linear1(x)
        x=self.linear2(x)
        return x

nn = cnn()
#print(nn)

#input = torch.ones((64,3,32,32))
#print(nn(input).shape)

#writer = SummaryWriter("../logs_seq")

loss = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(nn.parameters(), lr=0.02)

nn.train()
for epoch in range(100):
    tot_loss = 0
    tot_acc = 0
    tot_test = 0
    for data in dataloader:
        input_, target = data
        output = nn(input_)
        loss1 = loss(output, target)

        optimizer.zero_grad()
        loss1.backward()
        optimizer.step()

        tot_loss += loss1

    nn.eval()
    with torch.no_grad():
        for data in testloader:
            input_, target = data
            #print(input_.shape, target.shape)
            output = nn(input_)
            tot_acc += (output.argmax(1) == target).sum()
            tot_test += 64
            #print("total accu:",tot_acc)
        print("At epoch:{}, loss is {}, accuracy is {}".format(epoch, tot_loss, tot_acc/tot_test))


torch.save(nn,"cifar10_method.pth")






