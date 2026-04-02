import torch
import torch.nn as nn
import torch.nn.functional as F

class IDSCNNLSTM(nn.Module):
    """
    用于入侵检测的CNN-LSTM混合网络，结合CNN空间特征提取和LSTM时序分析能力
    
    参数:
        input_dim: 输入特征维度
        hidden_dim: LSTM隐藏层维度
        num_layers: LSTM层数
        dropout: Dropout比率
        bidirectional: 是否使用双向LSTM
        num_classes: 分类类别数，默认为2（二分类）
    """
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.5, 
                bidirectional=True, num_classes=2):
        super(IDSCNNLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        
        # CNN部分 - 1D卷积提取空间特征
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.25)
        
        # CNN部分 - 第二个卷积层
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(0.25)
        
        # 计算卷积后的特征维度
        self.cnn_output_dim = (input_dim // 4) * 128  # 经过两次池化，特征维度减小为原来的1/4
        
        # LSTM部分
        self.lstm = nn.LSTM(
            input_size=128,  # CNN的输出通道数
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # 注意力机制
        self.attention = SelfAttention(hidden_dim * self.num_directions)
        
        # 全连接层
        self.fc1 = nn.Linear(hidden_dim * self.num_directions, 128)
        self.relu = nn.ReLU()
        self.dropout_fc = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        # 将输入重塑为CNN所需的形状 [batch_size, channels, seq_len]
        x = x.unsqueeze(1)  # [batch_size, 1, seq_len]
        
        # CNN特征提取
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # 准备LSTM输入 - 将CNN输出转换为序列
        # [batch_size, channels, seq_len/4] -> [batch_size, seq_len/4, channels]
        x = x.transpose(1, 2)
        
        # 初始化LSTM隐藏状态
        h0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_dim).to(x.device)
        
        # LSTM处理
        output, (h_n, c_n) = self.lstm(x, (h0, c0))
        
        # 应用注意力机制
        attn_output = self.attention(output)
        
        # 全连接分类层
        x = self.relu(self.fc1(attn_output))
        x = self.dropout_fc(x)
        x = self.fc2(x)
        
        return x

class SelfAttention(nn.Module):
    """
    自注意力机制，用于捕获序列中的重要信息
    
    参数:
        hidden_dim: 隐藏层维度
    """
    def __init__(self, hidden_dim):
        super(SelfAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, lstm_output):
        # lstm_output: [batch_size, seq_len, hidden_dim]
        batch_size, seq_len, hidden_dim = lstm_output.size()
        
        # 计算注意力权重
        attn_weights = self.attention(lstm_output)  # [batch_size, seq_len, 1]
        attn_weights = torch.softmax(attn_weights, dim=1)  # 在序列维度上进行softmax
        
        # 应用注意力权重进行加权求和
        context = torch.bmm(attn_weights.transpose(1, 2), lstm_output)  # [batch_size, 1, hidden_dim]
        context = context.squeeze(1)  # [batch_size, hidden_dim]
        
        return context 