from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.autograd import Variable
from RNN import Rnn
from GRU import GruRNN
from CNN import CNN
from LSTM import LSTM
from CLSTM import CLSTM
from torchvision import datasets, models, transforms

labels_map = {
    "healthy": 0,
    "hyperkinetic dysphonia": 1,
    "hypokinetic dysphonia": 2,
    "reflux laryngitis": 3,
}
root_path = '../Data/VOICED/processed_data/'
# root_path = '../Data/VOICED/data/'

model_type = 'GRU'
# parameters
epochs = 10
batch_size = 64
lr = 0.01


# 自定义数据集
class VoiceData(Dataset):  # 继承Dataset
    def __init__(self, root_dir, transform=None):  # __init__是初始化该类的一些基础参数
        self.root_dir = root_dir  # 文件目录
        self.transform = transform  # 变换
        self.voices = os.listdir(self.root_dir)  # 目录里的所有文件

    def __len__(self):  # 返回整个数据集的大小
        return len(self.voices)

    def __getitem__(self, index):  # 根据索引index返回dataset[index]
        voice_index = self.voices[index]  # 根据索引index获取声音信息
        voice_path = os.path.join(self.root_dir, voice_index)  # 获取索引为index的声音路径名
        voice_data = np.loadtxt(voice_path)
        voice_data = np.reshape(voice_data, (128, 12))  # CNN (1, 128,12) MFCC+LSTM(128,12)
        label = labels_map[voice_path.split('/')[-1].split('.')[0]]
        sample = {'voice': voice_data, 'label': label}
        if self.transform:
            sample = self.transform(sample)  # 对样本进行变换
        return sample  # 返回该样本


if __name__ == '__main__':
    data = VoiceData(root_path)

    # dataloader = DataLoader(data, batch_size=128, shuffle=True)  # 使用DataLoader加载数据
    # for i_batch, batch_data in enumerate(dataloader):
    #     print(i_batch)  # 打印batch编号
    #     print(batch_data['voice'].size())  # 打印该batch里面矩阵的大小
    #     print(batch_data['label'])  # 打印该batch里面声音的标签

    train_size = int(0.8 * len(data))
    test_size = len(data) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(data, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=True)

    model = GruRNN()
    # 定义损失和优化器
    loss_func = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # 训练网络
    loss_count = []
    for epoch in range(epochs):
        for step, batch_data in enumerate(train_loader):
            batch_x = Variable(batch_data['voice'].to(torch.float32))  # torch.Size([128, 1, 28, 28])
            batch_y = Variable(batch_data['label'])  # torch.Size([128])
            # 获取最后输出
            out = model(batch_x)  # torch.Size([128,4])
            # 获取损失
            loss = loss_func(out, batch_y)
            # 使用优化器优化损失
            opt.zero_grad()  # 清空上一步残余更新参数值
            loss.backward()  # 误差反向传播，计算参数更新值
            opt.step()  # 将参数更新值施加到net的parmeters上
            if step % 20 == 0:
                loss_count.append(loss)
                print('{}:\t'.format(step), loss.item())
                torch.save(model, r'log_' + model_type)
            if step % 100 == 0:
                for data in test_loader:
                    test_x = Variable(data['voice'].to(torch.float32))
                    test_y = Variable(data['label'])
                    out = model(test_x)
                    # print('test_out:\t',torch.max(out,1)[1])
                    # print('test_y:\t',test_y)
                    accuracy = torch.max(out, 1)[1].numpy() == test_y.numpy()
                    print('accuracy:\t', accuracy.mean())
                    break
    plt.figure('PyTorch_' + model_type + '_Loss')
    plt.plot(loss_count, label='Loss')
    plt.legend()
    plt.show()

    # 测试网络
    model = torch.load(r'log_' + model_type)
    accuracy_sum = []
    for i, test_data in enumerate(train_loader):
        test_x = Variable(test_data['voice'].to(torch.float32))
        test_y = Variable(test_data['label'])
        out = model(test_x)
        # print('test_out:\t',torch.max(out,1)[1])
        # print('test_y:\t',test_y)
        accuracy = torch.max(out, 1)[1].numpy() == test_y.numpy()
        accuracy_sum.append(accuracy.mean())
        print('accuracy:\t', accuracy.mean())
    print('总准确率：\t', sum(accuracy_sum) / len(accuracy_sum))
    # 精确率图
    print('总准确率：\t', sum(accuracy_sum) / len(accuracy_sum))
    plt.figure('Accuracy')
    plt.plot(accuracy_sum, 'o', label='accuracy')
    plt.title('Pytorch_CNN_Accuracy')
    plt.legend()
    plt.show()
