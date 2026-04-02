import torch
import torch.nn as nn
import torch.nn.functional as F

class IDSConvNet(nn.Module):
    """
    用于入侵检测的卷积神经网络
    
    参数:
        input_dim: 输入特征维度
        num_classes: 分类类别数，默认为2（二分类）
    """
    def __init__(self, input_dim, num_classes=2):
        super(IDSConvNet, self).__init__()
        
        # 将输入特征重塑为2D形式以用于卷积操作
        # 我们将特征转换为(1, input_dim)的形状，相当于单通道的宽度为input_dim的1D数据
        
        # 第一个卷积块
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.25)
        
        # 第二个卷积块
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(0.25)
        
        # 第三个卷积块
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout(0.25)
        
        # 计算卷积层之后的特征维度
        # 初始维度: input_dim
        # 经过3次池化，每次降维一半: input_dim/(2^3) = input_dim/8
        self.feature_dim = (input_dim // 8) * 256  # 256是第三个卷积层的输出通道数
        
        # 全连接层
        self.fc1 = nn.Linear(self.feature_dim, 512)
        self.dropout4 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # 重塑输入以适应卷积操作
        # 从 [batch_size, input_dim] 到 [batch_size, 1, input_dim]
        x = x.unsqueeze(1)
        
        # 第一个卷积块
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # 第二个卷积块
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # 第三个卷积块
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        x = self.dropout3(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout4(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x 