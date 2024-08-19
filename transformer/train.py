import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from net import ViT
from torchvision import datasets, transforms

data_transform =transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])


#加载训练集
train_dataset = datasets.MNIST(root='./data', train=True, transform=data_transform, download=False)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)
#加载测试集
test_dataset = datasets.MNIST(root='./data', train=False, transform=data_transform, download=False)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1000, shuffle=True)

#有显卡的话，将模型数据转到GPU
device = "cuda" if torch.cuda.is_available () else 'cpu'
model = ViT(img_size=28, patch_size=7, in_chans=1, num_classes=10, embed_dim=64, depth=4, num_heads=8).to(device)
#随机梯度下降优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#定义训练函数
def train(dataloader, model, optimizer):
    model.train()
    loss, current, n = 0.0, 0.0, 0
    
    for batch, (data,y) in enumerate(dataloader):
    
        #前向传播
        data, y = data.to(device), y.to(device)
        output = F.log_softmax(model(data), dim=1)
        cur_loss = F.nll_loss(output, y)
        _, pred = torch.max(output, 1)
        cur_acc = torch.eq(y, pred).float().mean().item()

        #反向传播
        optimizer.zero_grad()
        cur_loss.backward()
        optimizer.step()
        loss += cur_loss.item()
        current += cur_acc
        n += 1

    train_loss = loss/n
    train_accuracy = current/n

    print("train loss:" + str(train_loss))
    print("train acc:" + str(train_accuracy))
    return train_loss, train_accuracy
    
#定义测试函数
def val(dataloader, model):
    model.eval()
    loss, current, n = 0.0, 0.0, 0
    with torch.no_grad():
       for batch, (data,y) in enumerate(dataloader):

        data, y = data.to(device), y.to(device)
        output = F.log_softmax(model(data), dim=1)
        cur_loss = F.nll_loss(output, y)
        _, pred = torch.max(output, 1)
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

for t in range(epoch):
   print(f'epoch{t}\n------------')
   train(train_loader, model, optimizer)
   val(test_loader, model)
   
print('Done!') 