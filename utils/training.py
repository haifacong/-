import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'SimHei'

def train_model(model, train_loader, val_loader, n_epochs=30, learning_rate=0.001, 
               device='cuda', patience=5, model_save_path=None):

    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
    }
    

    best_val_loss = float('inf')
    counter = 0
    

    start_time = time.time()
    
    # 训练循环
    for epoch in range(n_epochs):

        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_loop = tqdm(train_loader, desc=f'Epoch {epoch+1}/{n_epochs} [Train]')
        for inputs, targets in train_loop:

            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            # 统计
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()
            
            # 更新
            train_loop.set_postfix({'loss': loss.item(), 'acc': train_correct / train_total})
            
        # 计算训练损失和准确率
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct / train_total
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_loop = tqdm(val_loader, desc=f'Epoch {epoch+1}/{n_epochs} [Val]')
            for inputs, targets in val_loop:
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # 统计
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
                
                # 更新进度条
                val_loop.set_postfix({'loss': loss.item(), 'acc': val_correct / val_total})
                
        # 计算验证损失和准确率
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / val_total
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # 打印本轮训练结果
        print(f'Epoch {epoch+1}/{n_epochs} - '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        # 检查是否需要早停
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            # 保存最佳模型
            if model_save_path:
                torch.save(model.state_dict(), model_save_path)
        else:
            counter += 1
            if counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    # 加载最佳模型（如果有）
    if model_save_path:
        model.load_state_dict(torch.load(model_save_path))
    
    # 打印总训练时间
    total_time = time.time() - start_time
    print(f'训练完成，总耗时: {total_time:.2f} 秒')
    
    return model, history

def plot_training_history(history, save_path='training_history.png'):
    plt.figure(figsize=(12, 5))
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='训练损失')
    plt.plot(history['val_loss'], label='验证损失')
    plt.title('训练和验证损失')
    plt.xlabel('轮次')
    plt.ylabel('损失')
    plt.legend()
    
    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='训练准确率')
    plt.plot(history['val_acc'], label='验证准确率')
    plt.title('训练和验证准确率')
    plt.xlabel('轮次')
    plt.ylabel('准确率')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"训练历史图已保存到: {save_path}")
    plt.close()