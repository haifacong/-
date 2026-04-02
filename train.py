
import os
import argparse
import torch
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, random_split

from models import IDSConvNet, IDSLSTM, IDSCNNLSTM
from utils.data_utils import get_dataset_loader
from utils.training import train_model, plot_training_history
from utils.metrics import compute_metrics, print_metrics, evaluate_model

def main():
    parser = argparse.ArgumentParser(description='训练入侵检测模型')
    parser.add_argument('--dataset', type=str, default='nsl_kdd',
                        choices=['nsl_kdd', 'cicids2017'],
                        help='要使用的数据集名称')
    parser.add_argument('--model', type=str, default='cnn',
                        choices=['cnn', 'lstm', 'cnn_lstm'],
                        help='要训练的模型类型')
    parser.add_argument('--batch_size', type=int, default=64, 
                        help='批量大小')
    parser.add_argument('--epochs', type=int, default=30,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='学习率')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='隐藏层维度')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='层数')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout比率')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='数据目录')
    parser.add_argument('--save_dir', type=str, default='./saved_models',
                        help='模型保存目录')
    parser.add_argument('--no_cuda', action='store_true',
                        help='禁用CUDA')
    args = parser.parse_args()
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"使用设备: {device}")
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 数据集目录
    dataset_dir = os.path.join(args.data_dir, args.dataset)
    
    # 加载数据集
    print(f"正在加载 {args.dataset} 数据集...")
    if args.dataset.lower() == 'cicids2017':
        print("使用CICIDS2017数据集")
    elif args.dataset.lower() == 'nsl_kdd':
        print("使用NSL_KDD数据集")
    
    train_loader, test_loader, feature_dim = get_dataset_loader(
        args.dataset, dataset_dir, args.batch_size
    )
    print(f"特征维度: {feature_dim}")
    
    # 初始化模型
    print(f"初始化 {args.model.upper()} 模型...")
    if args.model == 'cnn':
        model = IDSConvNet(input_dim=feature_dim)
    elif args.model == 'lstm':
        model = IDSLSTM(
            input_dim=feature_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout
        )
    elif args.model == 'cnn_lstm':
        model = IDSCNNLSTM(
            input_dim=feature_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout
        )
    
    # 模型保存路径
    model_save_path = os.path.join(
        args.save_dir, 
        f"{args.model}_{args.dataset}_model.pth"
    )
    
    # 训练模型
    print("开始训练...")
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        n_epochs=args.epochs,
        learning_rate=args.lr,
        device=device,
        model_save_path=model_save_path
    )
    
    # 绘制训练历史
    plot_training_history(history)
    
    # 评估模型
    print("评估模型...")
    metrics = evaluate_model(model, test_loader, device)
    print_metrics(metrics)
    
    print(f"模型已保存到: {model_save_path}")

if __name__ == "__main__":
    main() 