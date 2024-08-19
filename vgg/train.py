import torch
import time
import numpy as np
from torch import nn
from net import vgg
from torchvision import datasets, transforms

data_transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])

#加载训练集
train_dataset = datasets.MNIST(root='./data', train=True, transform=data_transform, download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
#加载测试集
test_dataset = datasets.MNIST(root='./data', train=False, transform=data_transform, download=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=16, shuffle=True)

#有显卡的话，将模型数据转到GPU
device = "cuda" if torch.cuda.is_available () else 'cpu'
model = vgg().to(device)

#交叉熵损失函数
loss_fn = nn.CrossEntropyLoss()
#随机梯度下降优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#定义训练函数
def train(dataloader, model, loss_fn, optimizer):
    loss, current, n = 0.0, 0.0, 0
    start_time = time.time()
    for batch, (X,y) in enumerate(dataloader):
        #将单通道的图片改成三通道
        X = np.array(X)
        X = X.transpose((1,0,2,3))
        image = np.concatenate ((X,X,X),axis=0)
        image = image.transpose((1,0,2,3))
        image = torch.tensor(image) 

        #前向传播
        image, y = image.to(device), y.to(device)
        output = model(image)
        cur_loss = loss_fn(output,y)
        _, pred = torch.max(output, 1)
        cur_acc = torch.eq(y, pred).float().mean().item()
        #反向传播
        optimizer.zero_grad()
        cur_loss.backward()
        optimizer.step()
        loss += cur_loss.item()
        current += cur_acc
        n += 1
        end_time = time.time()
        elapsed_time = (end_time - start_time)/60
    train_loss = loss/n
    train_accuracy = current/n
    print(f"Epoch completed in {elapsed_time:.2f} minutes")
    print("train loss:" + str(train_loss))
    print("train acc:" + str(train_accuracy))      
    
#定义测试函数
def val(dataloader, model, loss_fn):
    model.eval()
    loss, current, n = 0.0, 0.0, 0
    with torch.no_grad():
       for batch, (X,y) in enumerate(dataloader):
        #将单通道的图片改成三通道
        X = np.array(X)
        X = X.transpose((1,0,2,3))
        image = np.concatenate ((X,X,X),axis=0)
        image = image.transpose((1,0,2,3))
        image = torch.tensor(image)

        image, y = image.to(device), y.to(device)
        output = model(image)
        cur_loss = loss_fn(output,y)
        _, pred = torch.max(output, axis=1)
        cur_acc = torch.eq(y, pred).float().mean().item()

        loss += cur_loss.item()
        current += cur_acc
        n += 1
    test_average_loss = loss/n
    test_average_accuracy = current/n
    print("test loss:" + str(test_average_loss))
    print("test acc:" + str(test_average_accuracy))

#开始训练
epoch = 5
for t in range(epoch):
   print(f'epoch{t}\n------------')
   train(train_loader, model, loss_fn, optimizer)
   val(test_loader, model, loss_fn)
   
print('Done!')   


