import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Sequential


class CnnNet(nn.Module):
    def __init__(self, batch_norm=False):
        super(CnnNet, self).__init__()
        # 确保输入通道数与Keras模型一致
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.global_avg_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32, 27)  # 根据输入尺寸计算线性层的输入特征数
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        # x = F.relu(x)  # 在PyTorch中，激活函数通常作为单独的函数调用
        x = self.conv2(x)
        x = F.relu(x)
        x = self.max_pool(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.max_pool(x)
        x = self.global_avg_pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        # x = self.dropout(x)
        x = F.log_softmax(x, dim=1)  # 通常使用log_softmax进行多分类任务
        return x


class CnnLstmNet(nn.Module):
    def __init__(self, batch_norm=False):
        super(CnnLstmNet, self).__init__()
        # 确保输入通道数与Keras模型一致
        self.conv1 = nn.Conv1d(in_channels=5, out_channels=128, kernel_size=5, stride=1)
        self.max_pool = nn.MaxPool1d(kernel_size=4, padding=2)
        self.lstm = nn.LSTM(input_size=1, hidden_size=25, batch_first=True)
        self.fc = nn.Linear(in_features=25, out_features=27)

    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool(x)
        r_out, (h_n, h_c) = self.lstm(x, None)
        out = self.fc(r_out[:, -1, :])  # 通常使用log_softmax进行多分类任务
        out = F.log_softmax(out, dim=1)
        return out


class LstmNet(nn.Module):
    def __init__(self, num_layers):
        super(LstmNet, self).__init__()
        # 确保输入通道数与Keras模型一致
        self.lstm = nn.LSTM(input_size=25, hidden_size=50, batch_first=True, dropout=0.2, num_layers=num_layers)
        self.fc = nn.Linear(in_features=50, out_features=27)

    def forward(self, x):
        r_out, (h_n, h_c) = self.lstm(x, None)
        out = self.fc(r_out[:, -1, :])
        out = F.log_softmax(out, dim=1)
        return out


class CnnGruNet(nn.Module):
    def __init__(self, batch_norm=False):
        super(CnnGruNet, self).__init__()
        # 确保输入通道数与Keras模型一致
        self.conv1 = nn.Conv1d(in_channels=5, out_channels=128, kernel_size=5, stride=1)
        self.max_pool = nn.MaxPool1d(kernel_size=4, padding=2)
        self.gru = nn.GRU(input_size=1, hidden_size=25, batch_first=True)
        self.fc = nn.Linear(in_features=25, out_features=27)

    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool(x)
        output, h_n = self.gru(x, None)
        out = self.fc(output[:, -1, :])
        out = F.log_softmax(out, dim=1)
        return out


class GruNet(nn.Module):
    def __init__(self, num_layers):
        super(GruNet, self).__init__()
        # 确保输入通道数与Keras模型一致
        self.gru = nn.GRU(input_size=25, hidden_size=50, batch_first=True, dropout=0.2, num_layers=num_layers)
        self.fc = nn.Linear(in_features=50, out_features=27)

    def forward(self, x):
        output, h_n = self.gru(x, None)
        out = self.fc(output[:, -1, :])
        out = F.log_softmax(out, dim=1)
        return out

# oldnet = OldNet()
# oldnet(torch.ones(10,1,5,5))
