##导入模块
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import torchvision
import torch.utils.data as Data
from torchvision import transforms

##导入数据MNIST手写数字数据集
train_data = torchvision.datasets.MNIST(root = "./data/MNIST",train = True,transform = transforms.ToTensor(),download = False)
train_loader = Data.DataLoader(dataset = train_data,batch_size = 64,shuffle = True,num_workers = 0)
test_data = torchvision.datasets.MNIST(root = "./data/MNIST",train = False,transform = transforms.ToTensor(),download = False)
test_loader = Data.DataLoader(dataset = test_data,batch_size = 64,shuffle = True,num_workers = 0)

##搭建RNN分类器
class RNNimc(nn.Module):
    def __init__(self,input_dim,hidden_dim,layer_dim,output_dim):
        """
        input_dim:输入数据的维度（图片每行的数据像素点）
        hidden_dim:RNN神经元个数
        layer_dim:RNN的层数
        output_dim:隐藏层的层数
        """
        super(RNNimc,self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.rnn = nn.RNN(input_dim,hidden_dim,layer_dim,batch_first = True,nonlinearity = 'relu')
        self.fc1 = nn.Linear(hidden_dim,output_dim)
    def forward(self,x):
        """ x:[batch,time_step,input_dim]
            time_step = 所有像素个数/input_dim
            out:[batch,time_step,output_size]
            h_n:[layer_dim,batch,hidden_dim]
        """
        out, h_n = self.rnn(x,None)
        out = self.fc1(out[:,-1,:])
        return out


##模型调用
input_dim = 28
hidden_dim = 128
layer_dim = 1
output_dim = 10
MyRNNimc = RNNimc(input_dim,hidden_dim,layer_dim,output_dim)
print(MyRNNimc)

##对模型进行训练
optimizer = torch.optim.RMSprop(MyRNNimc.parameters(),lr = 0.0003)
criterion = nn.CrossEntropyLoss()
train_loss_all = []
train_acc_all = []
test_loss_all = []
test_acc_all = []
num_epochs = 30
for epoch in range(num_epochs):
    print('Epoch{}/{}'.format(epoch,num_epochs-1))
    MyRNNimc.train()
    corrects = 0
    train_num = 0
    for step,(b_x,b_y) in enumerate(train_loader):
        xdata = b_x.view(-1,28,28)
        output = MyRNNimc(xdata)
        pre_lab = torch.argmax(output,1)
        loss = criterion(output,b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print(loss.item)
        loss += loss.item() * b_x.size(0)
        corrects += torch.sum(pre_lab == b_y.data)
        train_num += b_x.size(0)
    train_loss_all.append(loss/train_num)
    train_acc_all.append(corrects.double().item()/train_num)
    print('{}Train Loss:{:.4f} Train Acc:{:.4f}'.format(epoch,train_loss_all[-1],train_acc_all[-1]))
    MyRNNimc.eval()
    corrects = 0
    test_num = 0
    for step,(b_x,b_y) in enumerate(test_loader):
        xdata = b_x.view(-1,28,28)
        output = MyRNNimc(xdata)
        pre_lab = torch.argmax(output,1)
        loss = criterion(output,b_y)
        loss +=loss.item()*b_x.size(0)
        corrects += torch.sum(pre_lab == b_y.data)
        test_num +=b_x.size(0)
    test_loss_all.append(loss/test_num)
    test_acc_all.append(corrects.double().item()/test_num)
    print('{}Test Loss:{:.4f} Test Acc:{:.4f}'.format(epoch,test_loss_all[-1],test_acc_all[-1]))

train_loss_all_np = []
test_loss_all_np = []
# train_acc_all_np = []
# test_acc_all_np = []
for i in range(30):
    train_loss_all_np.append(train_loss_all[i].detach().numpy())
    #train_acc_all_np.append(train_acc_all[i].detach().numpy())
    test_loss_all_np.append(test_loss_all[i].detach().numpy())
    #test_acc_all_np.append(train_acc_all[i].detach().numpy())

##训练集损失和测试集损失&精度可视化
plt.figure(figsize=(14,5))
plt.subplot(1,2,1)
plt.plot(train_loss_all_np,"ro-",label = "Train Loss")
plt.plot(test_loss_all_np,"bs-",label = "Val Loss")
plt.legend()
plt.xlabel("epoch")
plt.ylabel("Loss")
plt.subplot(1,2,2)
plt.plot(train_acc_all,"ro-",label = "Train acc")
plt.plot(test_acc_all,"bs-",label = "Val acc")
plt.xlabel("epoch")
plt.ylabel("acc")
plt.legend()
plt.show()