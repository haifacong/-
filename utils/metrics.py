import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def compute_metrics(y_true, y_pred):
    # 确保输入是numpy数组
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
        
    # 评估指标
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='binary')
    rec = recall_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')
    
    # 混淆矩阵
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    metrics_dict = {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'true_positive': tp,
        'false_positive': fp,
        'true_negative': tn,
        'false_negative': fn
    }
    
    return metrics_dict

def print_metrics(metrics_dict):
    print("评估指标:")
    print(f"准确率 (Accuracy): {metrics_dict['accuracy']:.4f}")
    print(f"精确率 (Precision): {metrics_dict['precision']:.4f}")
    print(f"召回率 (Recall): {metrics_dict['recall']:.4f}")
    print(f"F1分数 (F1-score): {metrics_dict['f1_score']:.4f}")
    print("\n混淆矩阵:")
    print(f"真正例 (TP): {metrics_dict['true_positive']}")
    print(f"假正例 (FP): {metrics_dict['false_positive']}")
    print(f"真负例 (TN): {metrics_dict['true_negative']}")
    print(f"假负例 (FN): {metrics_dict['false_negative']}")
    
def evaluate_model(model, test_loader, device='cuda'):

    model.eval()
    
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # 前向传播
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            # 收集真实标签和预测标签
            y_true.extend(targets.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    
    # 计算评估指标
    metrics = compute_metrics(np.array(y_true), np.array(y_pred))
    
    return metrics 