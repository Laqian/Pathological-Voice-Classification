import torch


# 定义卷积网络结构
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1,
                            out_channels=16,
                            kernel_size=3,
                            stride=2,
                            padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU()
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, 3, 2, 1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU()
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, 3, 2, 1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, 2, 2, 0),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )
        self.mlp1 = torch.nn.Linear(8 * 1 * 64, 100)
        self.mlp2 = torch.nn.Linear(100, 4)

    def forward(self, x):
        x = self.conv1(x)  # torch.Size([128, 16, 64, 9])
        x = self.conv2(x)  # torch.Size([128, 32, 32, 5])
        x = self.conv3(x)  # torch.Size([128, 64, 16, 3])
        x = self.conv4(x)  # torch.Size([128, 64, 8, 1])
        x = self.mlp1(x.view(x.size(0), -1))
        x = self.mlp2(x)
        return x
