import torch
import time
import numpy as np
from torch import nn
from net import resnet18
from torchvision import datasets, transforms

data_transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])

#加载训练集
train_dataset = datasets.MNIST(root='./data', train=True, transform=data_transform, download=False)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
#加载测试集
test_dataset = datasets.MNIST(root='./data', train=False, transform=data_transform, download=False)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=True)

#有显卡的话，将模型数据转到GPU
device = "cuda" if torch.cuda.is_available () else 'cpu'
model = resnet18().to(device)

#交叉熵损失函数
loss_fn = nn.CrossEntropyLoss()
#随机梯度下降优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#定义训练函数
def train(dataloader, model, loss_fn, optimizer):
    model.train()
    loss, current, n = 0.0, 0.0, 0
    start_time = time.time()
    for batch, (X,y) in enumerate(dataloader):
     
        #前向传播
        image, y = X.to(device), y.to(device)
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
    return train_loss, train_accuracy
    
#定义测试函数
def val(dataloader, model, loss_fn):
    model.eval()
    loss, current, n = 0.0, 0.0, 0
    with torch.no_grad():
       for batch, (X,y) in enumerate(dataloader):

        image, y = X.to(device), y.to(device)
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
    return test_average_loss, test_average_accuracy

#开始训练
epoch = 10
best_val_loss = float('inf')  # 初始化最好的验证损失
patience = 3  # 早停法的容忍度
counter = 0  # 用于记录没有改进的epoch数量
for t in range(epoch):
   print(f'epoch{t}\n------------')
   train_loss, train_accuracy = train(train_loader, model, loss_fn, optimizer)
   test_average_loss, test_average_accuracy = val(test_loader, model, loss_fn)
   if test_average_loss < best_val_loss:
       best_val_loss = test_average_loss
       counter = 0  # 重置计数器
       # 可以在这里保存模型的状态
   else:
       counter += 1
       if counter >= patience:
           print("Early stopping triggered.")
           break  # 早停
   
print('Done!')   
