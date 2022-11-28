from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, models, transforms
import time
import os
import copy

if __name__ == "__main__":
    # 参数设置
    data_dir = "./data/ants_bees"  # 文件夹下有2个子文件train和val；每个文件夹下各有2个子文件ants和bees（两类）
    model_name = "resnet"  # 网络模型名
    num_classes = 2  # 类别数
    epochs = 15  # 训练epochs
    device = 'cpu'  # 训练方法：GPU or CPU
    input_size = 224  # 输入图片尺寸

    # 数据集设置
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),  # 随机裁剪然后resize
            transforms.RandomHorizontalFlip(),  # 随机水平翻转
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 归一化
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 归一化
        ])}
    image_sets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    data_loaders = {x: DataLoader(image_sets[x], batch_size=8, shuffle=True, num_workers=4) for x in ['train', 'val']}

    # 模型设置
    model_ft = models.resnet18(pretrained=True)  # resnet18模型
    for param in model_ft.parameters():  # param: 每一层权重值
        param.requires_grad = False  # 参数属性设置为不更新
    num_feats = model_ft.fc.in_features  # 提取的特征维度
    model_ft.fc = nn.Linear(num_feats, num_classes)  # 修改最后一层fc
    model_ft = model_ft.to(device)
    params_to_update = []
    for name, param in model_ft.named_parameters():  # name:层名;  param:权重值
        if param.requires_grad:
            params_to_update.append(param)  # 需要更新的权重参数,即最后一个全连接层:特征维度*类别数

    # 优化器和损失函数设置
    optimizer = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    since = time.time()
    best_model_wts = copy.deepcopy(model_ft.state_dict())  # 记录最优模型
    best_acc = 0.0
    for epoch in range(epochs):
        print('Epoch {}/{} -----------'.format(epoch, epochs - 1))
        for phase in ['train', 'val']:
            if phase == 'train':
                model_ft.train()
            else:
                model_ft.eval()
            running_loss = 0.0
            running_corrects = 0
            for inputs, labels in data_loaders[phase]:  # 按照batch取数据
                inputs = inputs.to(device)  # 输入数据
                labels = labels.to(device)  # 标签, labels = tensor([0, 1, 0, 0, 1, 0, 1, 1])
                optimizer.zero_grad()  # 零参数梯度
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model_ft(inputs)  # 提取特征
                    loss = criterion(outputs, labels)  # 计算交叉熵损失
                    _, predict = torch.max(outputs, 1)  # 预测label
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)  # 统计损失函数之和
                running_corrects += torch.sum(predict == labels.data)  # 计算预测正确样本个数
            epoch_loss = running_loss / len(data_loaders[phase].dataset)  # 计算所有样本的平均损失函数值
            epoch_acc = float(running_corrects) / len(data_loaders[phase].dataset)  # 计算所有样本预测的正确率
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))  # 打印结果
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model_ft.state_dict())  # 更新最优模型
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))  # 训练总时长
    print('Best val Acc: {:4f}'.format(best_acc))  # 打印最高测试准确率
    model_ft.load_state_dict(best_model_wts)  # load最好模型