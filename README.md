# IDS-QT — 基于深度学习的网络入侵检测系统

基于 PyTorch 的网络入侵检测系统，支持三种深度学习模型架构对网络流量进行二分类（正常/攻击），提供命令行（CLI）和 Django Web 两种操作界面。

## 功能特性

- **多模型架构**：支持 CNN、BiLSTM（带自注意力）、CNN-LSTM 混合模型
- **多数据集**：支持 NSL-KDD 和 CICIDS2017 数据集
- **实时流量捕获**：基于 Scapy 的网络流量实时采集，提取 28 维统计特征
- **攻击检测**：对捕获的流量数据进行在线威胁检测
- **可视化评估**：自动生成混淆矩阵、ROC 曲线、PR 曲线、训练历史图
- **Web 界面**：基于 Django 的现代暗色主题 Web UI，支持异步任务与实时日志

## 项目结构

```
IDS-QT/
├── main.py                    # CLI 入口
├── manage.py                  # Django 管理脚本
├── capture_to_csv.py          # 网络流量捕获工具
├── data_preprocessing.py      # 数据预处理脚本
├── train.py                   # 模型训练脚本
├── evaluate.py                # 模型评估脚本
├── requirements.txt           # Python 依赖
│
├── models/                    # 深度学习模型
│   ├── cnn_model.py           # IDSConvNet — 1D CNN（3 卷积块）
│   ├── lstm_model.py          # IDSLSTM — 双向 LSTM + 自注意力
│   └── cnn_lstm_model.py      # IDSCNNLSTM — CNN-BiLSTM 混合 + 注意力
│
├── utils/                     # 工具模块
│   ├── data_utils.py          # 数据加载与预处理
│   ├── training.py            # 训练流程（早停、模型保存）
│   └── metrics.py             # 评估指标
│
├── ids_web/                   # Django 项目配置
│   ├── settings.py
│   └── urls.py
│
├── ids/                       # Django 应用
│   ├── views.py               # API 端点
│   ├── tasks.py               # 后台任务管理器
│   ├── templates/ids/
│   │   └── index.html         # Web UI 主页
│   └── static/ids/
│       ├── css/style.css      # 样式（暗色主题）
│       └── js/app.js          # 前端逻辑
│
├── data/                      # 数据集目录
│   ├── nsl_kdd/               # NSL-KDD 数据集
│   └── cicids2017/            # CICIDS2017 数据集
│
├── saved_models/              # 训练好的模型权重
├── results/                   # 评估结果（图表）
└── captured_data/             # 捕获的流量数据
```

## 环境要求

- Python 3.7+
- PyTorch 1.7+
- CUDA（可选，用于 GPU 加速）
- 管理员/root 权限（实时流量捕获需要）

## 快速开始

### 1. 安装依赖

```bash
# 创建并激活虚拟环境（推荐）
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据预处理

```bash
python main.py --task preprocess --dataset cicids2017
```

### 3. 训练模型

```bash
# 训练单个模型
python main.py --task train --dataset cicids2017 --model cnn_lstm --epochs 30

# 训练所有模型
python main.py --task train --dataset cicids2017 --model all
```

### 4. 评估模型

```bash
python main.py --task evaluate --dataset cicids2017 --model cnn
```

### 5. 完整流水线（预处理 + 训练 + 评估）

```bash
python main.py --task all --dataset cicids2017 --model all
```

## 实时流量检测

```bash
# 捕获网络流量（需要管理员权限，单位：秒）
python main.py --task capture --capture_time 60

# 对捕获的流量进行攻击检测
python main.py --task detect --capture_file captured_data/file.csv --detect_model cnn_lstm
```

> **注意**：CICIDS2017 训练的模型与实时捕获的流量特征匹配；NSL-KDD 模型可能与实时流量特征不对齐。

## Web 界面

```bash
python manage.py runserver
```

访问 `http://127.0.0.1:8000` 即可使用 Web 界面，功能包括：

- 数据集选择与预处理
- 模型训练（可配置超参数）
- 模型评估与可视化
- 实时流量捕获
- 在线攻击检测
- 实时日志输出与导出

## CLI 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--task` | — | 任务类型：`preprocess` / `train` / `evaluate` / `all` / `capture` / `detect` |
| `--dataset` | `cicids2017` | 数据集：`nsl_kdd` / `cicids2017` |
| `--model` | `cnn_lstm` | 模型：`cnn` / `lstm` / `cnn_lstm` / `all` |
| `--batch_size` | `64` | 批大小 |
| `--epochs` | `30` | 训练轮数 |
| `--lr` | `0.001` | 学习率 |
| `--hidden_dim` | `128` | LSTM 隐藏维度 |
| `--num_layers` | `2` | LSTM 层数 |
| `--no_cuda` | `False` | 禁用 GPU |
| `--capture_time` | `60` | 流量捕获时长（秒） |

## 模型架构

### IDSConvNet（CNN）

3 层 1D 卷积块（64→128→256），每块包含 BatchNorm + ReLU + MaxPool + Dropout，后接 3 层全连接（512→128→2）。

### IDSLSTM（BiLSTM + Attention）

双向 LSTM + 自注意力机制。注意力层通过 Linear→Tanh→Linear→Softmax 计算序列各时间步的重要性权重，加权求和后输入全连接分类层。

### IDSCNNLSTM（混合模型）

CNN 特征提取（2 卷积块）→ BiLSTM 序列建模 → 自注意力 → 全连接分类。融合 CNN 的局部特征提取能力与 LSTM 的时序建模能力。

## 训练策略

- **优化器**：Adam
- **损失函数**：CrossEntropyLoss
- **早停机制**：验证损失连续 5 个 epoch 不下降则停止
- **模型保存**：基于验证损失的最优模型自动保存

## 技术栈

| 类别 | 技术 |
|------|------|
| 深度学习 | PyTorch |
| 数据处理 | NumPy, Pandas, scikit-learn |
| 流量捕获 | Scapy |
| 可视化 | Matplotlib, Seaborn |
| Web 框架 | Django |
| 前端 | HTML/CSS/JavaScript |
