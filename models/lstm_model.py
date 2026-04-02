import torch
import torch.nn as nn

class IDSLSTM(nn.Module):
    """
    用于入侵检测的LSTM网络
    
    参数:
        input_dim: 输入特征维度
        hidden_dim: 隐藏层维度
        num_layers: LSTM层数
        dropout: Dropout比率
        bidirectional: 是否使用双向LSTM
        num_classes: 分类类别数，默认为2（二分类）
    """
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.5, 
                bidirectional=True, num_classes=2):
        super(IDSLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=1,  # 每个时间步输入一个特征
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
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # 将输入重塑为序列形式 [batch_size, seq_len, input_size]
        # 我们将每个特征视为序列中的一个时间步
        batch_size = x.size(0)
        seq_len = x.size(1)
        x = x.view(batch_size, seq_len, 1)
        
        # 初始化隐藏状态和单元状态
        h0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_dim).to(x.device)
        
        # LSTM前向传播
        # output: [batch_size, seq_len, hidden_dim * num_directions]
        output, (h_n, c_n) = self.lstm(x, (h0, c0))
        
        # 应用注意力机制
        attn_output = self.attention(output)
        
        # 全连接层
        x = self.relu(self.fc1(attn_output))
        x = self.dropout(x)
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