import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import subprocess
import pandas as pd
from capture_to_csv import capture_and_save_to_csv
import warnings
warnings.filterwarnings("ignore")

def main():
    parser = argparse.ArgumentParser(description='深度学习入侵检测系统')
    parser.add_argument('--task', type=str, required=True,
                        choices=['preprocess', 'train', 'evaluate', 'all', 'capture', 'detect'],
                        help='要执行的任务')
    parser.add_argument('--dataset', type=str, default='nsl_kdd',
                        choices=['nsl_kdd', 'cicids2017'],
                        help='要使用的数据集名称')
    parser.add_argument('--model', type=str, default='cnn',
                        choices=['cnn', 'lstm', 'cnn_lstm', 'all'],
                        help='要训练/评估的模型类型')
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
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='数据目录')
    parser.add_argument('--save_dir', type=str, default='./saved_models',
                        help='模型保存目录')
    parser.add_argument('--results_dir', type=str, default='./results',
                        help='结果保存目录')
    parser.add_argument('--no_cuda', action='store_true',
                        help='禁用CUDA')
    parser.add_argument('--capture_time', type=int, default=60,
                        help='捕获网络流量的时间（秒）')
    parser.add_argument('--capture_file', type=str, default=None,
                        help='捕获的流量文件路径')
    parser.add_argument('--detect_model', type=str, default='cnn',
                        choices=['cnn', 'lstm', 'cnn_lstm'],
                        help='用于检测的模型类型')
    args = parser.parse_args()
    
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    
    if args.task == 'preprocess' or args.task == 'all':
        print("="*50)
        print(f"开始预处理 {args.dataset} 数据集...")
        preprocess_cmd = f"python data_preprocessing.py --dataset {args.dataset} --data_dir {args.data_dir}"
        # 对于CICIDS2017，添加preprocess_only选项，因为不需要下载
        if args.dataset == 'cicids2017':
            preprocess_cmd += " --preprocess_only"
        print(f"执行命令: {preprocess_cmd}")
        subprocess.run(preprocess_cmd, shell=True)
    
    if args.task == 'train' or args.task == 'all':
        print("="*50)
        print(f"开始训练模型...")
        
        models_to_train = []
        if args.model == 'all':
            models_to_train = ['cnn', 'lstm', 'cnn_lstm']
        else:
            models_to_train = [args.model]
        
        for model in models_to_train:
            print(f"训练{model.upper()}模型...")
            train_cmd = (
                f"python train.py --dataset {args.dataset} --model {model} "
                f"--batch_size {args.batch_size} --epochs {args.epochs} --lr {args.lr} "
                f"--hidden_dim {args.hidden_dim} --num_layers {args.num_layers} "
                f"--data_dir {args.data_dir} --save_dir {args.save_dir}"
            )
            
            if args.no_cuda:
                train_cmd += " --no_cuda"
                
            print(f"执行命令: {train_cmd}")
            subprocess.run(train_cmd, shell=True)
    
    if args.task == 'evaluate' or args.task == 'all':
        print("="*50)
        print(f"开始评估模型...")
        
        models_to_evaluate = []
        if args.model == 'all':
            models_to_evaluate = ['cnn', 'lstm', 'cnn_lstm']
        else:
            models_to_evaluate = [args.model]
        
        for model in models_to_evaluate:
            print(f"评估{model.upper()}模型...")
            eval_cmd = (
                f"python evaluate.py --dataset {args.dataset} --model {model} "
                f"--batch_size {args.batch_size} --hidden_dim {args.hidden_dim} "
                f"--num_layers {args.num_layers} --data_dir {args.data_dir} "
                f"--save_dir {args.results_dir}"
            )
            
            if args.no_cuda:
                eval_cmd += " --no_cuda"
                
            print(f"执行命令: {eval_cmd}")
            subprocess.run(eval_cmd, shell=True)
    
    # 捕获主机流量
    if args.task == 'capture':
        print("="*50)
        print(f"开始捕获主机流量，持续 {args.capture_time} 秒...")
        
        if args.capture_file is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(args.data_dir, f"captured_flows_{timestamp}.csv")
        else:
            output_path = args.capture_file
        
        captured_file = capture_and_save_to_csv(
            duration=args.capture_time,
            output_path=output_path,
            debug=True
        )
        
        if captured_file:
            print(f"流量捕获完成，数据已保存到: {captured_file}")
        else:
            print("流量捕获失败或未捕获到有效流量")
    
    # 识别流量
    if args.task == 'detect':
        print("="*50)
        print(f"开始识别流量...")
        
        # 检查捕获文件是否存在
        if args.capture_file is None:
            print("错误: 未指定捕获文件路径，请使用 --capture_file 参数指定")
            return
        
        if not os.path.exists(args.capture_file):
            print(f"错误: 捕获文件 {args.capture_file} 不存在")
            return
        
        from utils.data_utils import load_captured_traffic
        X, preprocessor = load_captured_traffic(args.capture_file)
        
        print(f"加载 {args.detect_model.upper()} 模型...")
        model_path = os.path.join(args.save_dir, f"{args.detect_model}_{args.dataset}_model.pth")
        
        if not os.path.exists(model_path):
            print(f"错误: 模型文件 {model_path} 不存在")
            return

        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        print(f"使用设备: {device}")
        
        from models import IDSConvNet, IDSLSTM, IDSCNNLSTM
        
        feature_dim = X.shape[1]
        
        # 初始化模型
        if args.detect_model == 'cnn':
            model = IDSConvNet(input_dim=feature_dim)
        elif args.detect_model == 'lstm':
            model = IDSLSTM(input_dim=feature_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers)
        elif args.detect_model == 'cnn_lstm':
            model = IDSCNNLSTM(input_dim=feature_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers)
        
        # 加载模型权重
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        # 转换输入数据为张量
        X_tensor = torch.FloatTensor(X).to(device)
        
        # 预测
        with torch.no_grad():
            outputs = model(X_tensor)
            _, predicted = torch.max(outputs, 1)
        
        # 保存结果
        results = pd.DataFrame({
            'prediction': predicted.cpu().numpy(),
            'is_attack': predicted.cpu().numpy() > 0  # 二分类：0=正常，1=攻击
        })
        
        # 统计结果
        attack_count = results['is_attack'].sum()
        total = len(results)
        attack_percentage = (attack_count / total) * 100 if total > 0 else 0
        
        print(f"检测到的总流量数: {total}")
        print(f"正常流量数: {total - attack_count}")
        print(f"攻击流量数: {attack_count}")
        print(f"攻击比例: {attack_percentage:.2f}%")
        
        # 保存结果到CSV文件
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        result_file = os.path.join(args.results_dir, f"detection_result_{timestamp}.csv")
        results.to_csv(result_file, index=False)
        print(f"检测结果已保存到: {result_file}")
        
        # 生成结果图表
        plt.figure(figsize=(10, 6))
        plt.bar(['正常流量', '攻击流量'], [total - attack_count, attack_count])
        plt.title('流量检测结果')
        plt.ylabel('流量数量')
        plt.grid(axis='y', alpha=0.3)
        
        for i, v in enumerate([total - attack_count, attack_count]):
            plt.text(i, v + 0.1, str(v), ha='center')
        
        plt.tight_layout()
        result_img = os.path.join(args.results_dir, f"detection_result_{timestamp}.png")
        plt.savefig(result_img)
        print(f"检测结果图表已保存到: {result_img}")
        
        return results, result_file, result_img

if __name__ == "__main__":
    main() 