import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
from torch.utils.data import DataLoader, TensorDataset

from models import IDSConvNet, IDSLSTM, IDSCNNLSTM
from utils.data_utils import get_dataset_loader
from utils.metrics import compute_metrics, print_metrics, evaluate_model

def plot_confusion_matrix(y_true, y_pred, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('混淆矩阵')
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"混淆矩阵已保存到: {save_path}")
    
    plt.close()

def plot_roc_curve(y_true, y_score, save_path=None):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假阳性率')
    plt.ylabel('真阳性率')
    plt.title('ROC曲线')
    plt.legend(loc="lower right")
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"ROC曲线已保存到: {save_path}")
    
    plt.close()
    
    return roc_auc

def plot_precision_recall_curve(y_true, y_score, save_path=None):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    avg_precision = np.mean(precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'精确率-召回率曲线 (AP = {avg_precision:.4f})')
    plt.xlabel('召回率')
    plt.ylabel('精确率')
    plt.title('精确率-召回率曲线')
    plt.legend(loc="lower left")
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"精确率-召回率曲线已保存到: {save_path}")
    
    plt.close()
    
    return avg_precision

def main():
    parser = argparse.ArgumentParser(description='评估入侵检测模型')
    parser.add_argument('--dataset', type=str, default='nsl_kdd',
                        choices=['nsl_kdd', 'cicids2017'],
                        help='要使用的数据集名称')
    parser.add_argument('--model', type=str, default='cnn',
                        choices=['cnn', 'lstm', 'cnn_lstm'],
                        help='要评估的模型类型')
    parser.add_argument('--batch_size', type=int, default=64, 
                        help='批量大小')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='隐藏层维度')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='层数')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='数据目录')
    parser.add_argument('--model_path', type=str, default=None,
                        help='模型路径')
    parser.add_argument('--save_dir', type=str, default='./results',
                        help='结果保存目录')
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
    
    _, test_loader, feature_dim = get_dataset_loader(
        args.dataset, dataset_dir, args.batch_size
    )
    
    # 初始化模型
    print(f"初始化 {args.model.upper()} 模型...")
    if args.model == 'cnn':
        model = IDSConvNet(input_dim=feature_dim)
    elif args.model == 'lstm':
        model = IDSLSTM(
            input_dim=feature_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers
        )
    elif args.model == 'cnn_lstm':
        model = IDSCNNLSTM(
            input_dim=feature_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers
        )
    
    # 加载模型权重
    model_path = args.model_path
    if model_path is None:
        model_path = f"./saved_models/{args.model}_{args.dataset}_model.pth"
    
    if os.path.exists(model_path):
        print(f"从 {model_path} 加载模型权重...")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"错误: 未找到模型文件 {model_path}")
        return
    
    model = model.to(device)
    model.eval()
    
    y_true = []
    y_pred = []
    y_score = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            scores = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            
            y_true.extend(targets.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_score.extend(scores)
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_score = np.array(y_score)
    
    # 计算并打印指标
    metrics = compute_metrics(y_true, y_pred)
    print_metrics(metrics)
    
    # 保存评估结果图像
    confusion_matrix_path = os.path.join(args.save_dir, "confusion_matrix.png")
    roc_curve_path = os.path.join(args.save_dir, "roc_curve.png")
    pr_curve_path = os.path.join(args.save_dir, "pr_curve.png")
    
    # 绘制并保存混淆矩阵
    plot_confusion_matrix(y_true, y_pred, save_path=confusion_matrix_path)
    
    # 绘制并保存ROC曲线
    roc_auc = plot_roc_curve(y_true, y_score, save_path=roc_curve_path)
    print(f"ROC AUC: {roc_auc:.4f}")
    
    # 绘制并保存精确率-召回率曲线
    avg_precision = plot_precision_recall_curve(y_true, y_score, save_path=pr_curve_path)
    print(f"平均精确率: {avg_precision:.4f}")
    
    print(f"评估结果已保存到: {args.save_dir}")

if __name__ == "__main__":
    main() 