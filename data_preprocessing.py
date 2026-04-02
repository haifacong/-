import os
import argparse
import pandas as pd
import numpy as np
import urllib.request
import gzip
import shutil
from sklearn.model_selection import train_test_split
from utils.data_utils import load_nsl_kdd, load_cicids2017
import warnings
warnings.filterwarnings("ignore")

def download_nsl_kdd(data_dir):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # 这里应该使用NSL_KDD的下载链接和处理方法
    # 由于原项目可能没有NSL_KDD的具体实现，这里只提供一个框架
    print("NSL_KDD数据集需要手动下载并放置到以下目录：", data_dir)
    print("请参考项目文档获取下载地址和处理方法")

def preprocess_dataset(dataset_name, data_dir, save_processed=True):
    print(f"正在预处理 {dataset_name} 数据集...")
    
    if dataset_name.lower() == 'nsl_kdd':
        X_train, X_test, y_train, y_test, preprocessor = load_nsl_kdd(data_dir)
    elif dataset_name.lower() == 'cicids2017':
        X_train, X_test, y_train, y_test, preprocessor = load_cicids2017(data_dir)
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")
    
    # 打印数据集信息
    print(f"训练集形状: {X_train.shape}")
    print(f"测试集形状: {X_test.shape}")
    print(f"训练集标签分布:\n{np.bincount(y_train)}")
    print(f"测试集标签分布:\n{np.bincount(y_test)}")
    
    # 保存处理后的数据（可选）
    if save_processed:
        processed_dir = os.path.join(data_dir, f"{dataset_name}_processed")
        if not os.path.exists(processed_dir):
            os.makedirs(processed_dir)
        
        np.save(os.path.join(processed_dir, "X_train.npy"), X_train)
        np.save(os.path.join(processed_dir, "X_test.npy"), X_test)
        np.save(os.path.join(processed_dir, "y_train.npy"), y_train)
        np.save(os.path.join(processed_dir, "y_test.npy"), y_test)
        
        print(f"处理后的数据已保存到 {processed_dir}")
    
    print("预处理完成！")

def main():
    parser = argparse.ArgumentParser(description='数据集下载和预处理')
    parser.add_argument('--dataset', type=str, default='nsl_kdd', 
                        choices=['nsl_kdd', 'cicids2017'],
                        help='要下载和预处理的数据集名称')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='数据存储目录')
    parser.add_argument('--download_only', action='store_true',
                        help='仅下载数据集，不进行预处理')
    parser.add_argument('--preprocess_only', action='store_true',
                        help='仅预处理数据集，不下载')
    args = parser.parse_args()
    
    dataset_dir = os.path.join(args.data_dir, args.dataset)
    os.makedirs(dataset_dir, exist_ok=True)
    
    # 下载数据集
    if not args.preprocess_only:
        print(f"正在下载 {args.dataset} 数据集...")
        if args.dataset == 'nsl_kdd':
            download_nsl_kdd(dataset_dir)
    
    # 预处理数据集
    if not args.download_only:
        preprocess_dataset(args.dataset, dataset_dir)

if __name__ == "__main__":
    main() 