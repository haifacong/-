# IDS-QT 项目数据流与 API 接口文档

---

## 一、项目文件结构总览

```
IDS-QT/
├── main.py                    # CLI 总入口（调度各子命令）
├── data_preprocessing.py      # 数据预处理脚本
├── train.py                   # 模型训练脚本
├── evaluate.py                # 模型评估脚本
├── capture_to_csv.py          # 实时流量捕获（Scapy）
├── models/
│   ├── cnn_model.py           # IDSConvNet 模型定义
│   ├── lstm_model.py          # IDSLSTM 模型定义
│   └── cnn_lstm_model.py      # IDSCNNLSTM 模型定义
├── utils/
│   ├── data_utils.py          # 数据加载与预处理工具
│   ├── training.py            # 训练循环与训练历史绘图
│   └── metrics.py             # 评估指标计算
├── ids/                       # Django 应用
│   ├── urls.py                # API 路由定义
│   ├── views.py               # API 视图函数
│   ├── tasks.py               # 后台任务管理器（TaskManager）
│   ├── templates/ids/index.html  # 前端页面
│   └── static/ids/
│       ├── css/style.css      # 前端样式
│       └── js/app.js          # 前端交互逻辑
└── ids_web/
    └── urls.py                # Django 项目根路由
```

---

## 二、整体数据流架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                        前端 (index.html + app.js)                    │
│                                                                     │
│  用户操作 → IDS.xxx() → fetch(POST /api/xxx/) → 轮询 /api/task_status/ │
│                                                                     │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ HTTP JSON
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     Django 后端 (ids/views.py)                       │
│                                                                     │
│  api_xxx() → 解析参数 → 构建 shell 命令 → task_manager.start_task()   │
│                                                                     │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ subprocess (shell)
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│              Python 脚本层 (main.py / train.py / evaluate.py 等)     │
│                                                                     │
│  加载数据(data_utils) → 构建模型(models/) → 训练/评估(utils/) → 输出   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 三、各文件间的数据传递关系

### 3.1 前端 → Django 后端

| 前端函数 | 请求目标 | 传递的参数 |
|---------|---------|-----------|
| `IDS.preprocess()` | `POST /api/preprocess/` | `{dataset, data_dir}` |
| `IDS.train()` | `POST /api/train/` | `{dataset, model, batch_size, epochs, lr, hidden_dim, num_layers, no_cuda}` |
| `IDS.evaluate()` | `POST /api/evaluate/` | `{dataset, model, batch_size, hidden_dim, num_layers, no_cuda, data_dir}` |
| `IDS.capture()` | `POST /api/capture/` | `{capture_time, data_dir}` |
| `IDS.detect()` | `POST /api/detect/` | `{dataset, model, no_cuda, data_dir}` |
| 轮询 | `GET /api/task_status/?since=N` | URL 参数 `since`（日志偏移量） |

前端通过 `IDS.getParams()` 统一从 DOM 表单元素收集参数，然后由各任务函数挑选所需字段传给后端。

### 3.2 Django 后端 → Python 脚本层

**views.py** 收到前端 POST 请求后，将参数拼接为 shell 命令字符串，交给 **TaskManager** 执行：

| 视图函数 | 构建的命令 | 调用的脚本 |
|---------|-----------|-----------|
| `api_preprocess()` | `python data_preprocessing.py --dataset {dataset} --data_dir {data_dir} [--preprocess_only]` | `data_preprocessing.py` |
| `api_train()` | `python train.py --dataset {dataset} --model {model} --batch_size ... --epochs ... --lr ... --hidden_dim ... --num_layers ... [--no_cuda]` | `train.py` |
| `api_evaluate()` | `python evaluate.py --dataset {dataset} --model {model} --batch_size ... --hidden_dim ... --num_layers ... --data_dir ... --save_dir {results_dir} [--no_cuda]` | `evaluate.py` |
| `api_capture()` | `python main.py --task capture --capture_time {time} --capture_file {file}` | `main.py` → `capture_to_csv.py` |
| `api_detect()` | `python main.py --task detect --capture_file {file} --detect_model {model} --dataset {dataset} [--no_cuda]` | `main.py`（内联检测逻辑） |

### 3.3 TaskManager 工作机制 (`ids/tasks.py`)

```
TaskManager（全局单例 task_manager）
│
├── start_task(task_name, command, on_complete)
│   ├── 检查 is_running（同一时刻只允许一个任务）
│   ├── 清空 log_lines、重置状态
│   └── 启动 daemon 线程 → _run_command()
│
├── _run_command(command, on_complete)
│   ├── subprocess.Popen(command, shell=True, stdout=PIPE)
│   ├── 逐行读取 stdout → 追加到 log_lines[]
│   ├── process.wait() → 根据 returncode 设置 status
│   └── 调用 on_complete() 回调（设置 result_data）
│
├── get_status() → 返回 {is_running, status, status_text, task_name, log_count, result_data}
├── get_logs(since) → 返回 log_lines[since:]
└── clear_logs() → 重置所有状态
```

**关键数据流：**
- 子进程的 stdout 输出 → `log_lines[]` → 前端轮询 `api_task_status` 获取
- 任务完成后的 `on_complete` 回调在 `result_data` 字典中写入结果文件路径
- 前端收到 `result_data` 后触发相应的数据加载（图片/表格）

### 3.4 Python 脚本层内部数据传递

#### 3.4.1 数据预处理流程

```
data_preprocessing.py
  │
  ├── preprocess_dataset(dataset_name, data_dir)
  │     │
  │     ├── [NSL-KDD]  → data_utils.load_nsl_kdd(data_dir)
  │     │     ├── 读取 KDDTrain+.txt / KDDTest+.txt
  │     │     ├── ColumnTransformer: 数值列→StandardScaler, 分类列→OneHotEncoder
  │     │     └── 返回 (X_train, X_test, y_train, y_test, preprocessor)
  │     │
  │     └── [CICIDS2017] → data_utils.load_cicids2017(data_dir)
  │           ├── 读取目录下所有 CSV 文件 → 合并
  │           ├── 提取 25 个有意义特征
  │           ├── 处理 inf/NaN/异常值(IQR)
  │           ├── StandardScaler 标准化
  │           └── 返回 (X_train, X_test, y_train, y_test, preprocessor)
  │
  └── 保存处理后的 numpy 文件:
        data/{dataset}_processed/X_train.npy
        data/{dataset}_processed/X_test.npy
        data/{dataset}_processed/y_train.npy
        data/{dataset}_processed/y_test.npy
```

#### 3.4.2 模型训练流程

```
train.py
  │
  ├── data_utils.get_dataset_loader(dataset, dataset_dir, batch_size)
  │     ├── 调用 load_nsl_kdd() 或 load_cicids2017()
  │     ├── 稀疏矩阵 → dense (toarray)
  │     ├── numpy → torch.FloatTensor / LongTensor
  │     ├── 构建 TensorDataset → DataLoader
  │     └── 返回 (train_loader, test_loader, feature_dim)
  │
  ├── 根据 model 参数实例化模型:
  │     ├── cnn     → IDSConvNet(input_dim=feature_dim)
  │     ├── lstm    → IDSLSTM(input_dim, hidden_dim, num_layers, dropout)
  │     └── cnn_lstm → IDSCNNLSTM(input_dim, hidden_dim, num_layers, dropout)
  │
  ├── training.train_model(model, train_loader, val_loader, ...)
  │     ├── Adam 优化器 + CrossEntropyLoss
  │     ├── 训练循环（带 tqdm 进度条 → 输出到 stdout → 被 TaskManager 捕获）
  │     ├── Early Stopping (patience=5)
  │     ├── 保存最佳模型 → saved_models/{model}_{dataset}_model.pth
  │     └── 返回 (model, history{train_loss, val_loss, train_acc, val_acc})
  │
  ├── training.plot_training_history(history)
  │     └── 保存 training_history.png（损失曲线 + 准确率曲线）
  │
  └── metrics.evaluate_model(model, test_loader, device)
        └── 返回 metrics_dict → print_metrics() 输出到 stdout
```

#### 3.4.3 模型评估流程

```
evaluate.py
  │
  ├── data_utils.get_dataset_loader() → (train_loader, test_loader, feature_dim)
  ├── 加载模型权重 ← saved_models/{model}_{dataset}_model.pth
  ├── 推理 test_loader → 收集 y_true, y_pred, y_score
  │
  ├── metrics.compute_metrics(y_true, y_pred) → metrics_dict
  ├── plot_confusion_matrix() → results/{model}_{dataset}/confusion_matrix.png
  ├── plot_roc_curve()        → results/{model}_{dataset}/roc_curve.png
  └── plot_precision_recall_curve() → results/{model}_{dataset}/pr_curve.png
```

#### 3.4.4 流量捕获流程

```
main.py --task capture
  │
  └── capture_to_csv.capture_and_save_to_csv(duration, output_path)
        │
        ├── FlowManager（管理所有活跃流）
        │     ├── process_packet(packet)
        │     │     ├── 从 Scapy 包提取 IP/TCP/UDP 五元组
        │     │     ├── 匹配已有 Flow 或创建新 Flow
        │     │     └── Flow.add_packet() 更新统计计数
        │     └── clean_expired_flows()（后台定时清理线程）
        │
        ├── scapy.sniff(prn=callback, timeout=duration)
        │
        ├── 遍历所有 Flow → Flow.extract_features()
        │     └── 提取 28 个统计特征（与 CICIDS2017 特征对齐）
        │
        └── pd.DataFrame → 保存 CSV
              → data/capture_flows/captured_flows.csv
```

#### 3.4.5 威胁检测流程

```
main.py --task detect
  │
  ├── data_utils.load_captured_traffic(capture_file)
  │     ├── 读取 CSV → 选择 26 个有意义特征
  │     ├── 处理 inf/NaN/异常值
  │     ├── StandardScaler 标准化
  │     └── 返回 (X_processed, preprocessor)
  │
  ├── 加载模型 ← saved_models/{model}_{dataset}_model.pth
  ├── X → torch.FloatTensor → model(X) → predicted
  │
  └── 保存结果:
        ├── results/detection_result_{timestamp}.csv (prediction, is_attack)
        └── results/detection_result_{timestamp}.png (柱状图)
```

### 3.5 Django 后端 → 前端（结果回传）

任务完成后，前端通过额外接口拉取结果数据：

| 接口 | 触发条件 | 数据来源 | 返回内容 |
|------|---------|---------|---------|
| `GET /api/result_image/{name}/` | 用户点击可视化按钮或任务完成自动触发 | `results/{model}_{dataset}/` 下的 PNG 文件 | 图片二进制流 |
| `GET /api/capture_data/` | 流量捕获任务完成 | `data/capture_flows/captured_flows.csv` | `{columns, data(前100行), total}` |
| `GET /api/detection_result/` | 威胁检测任务完成 | `results/detection_result_*.csv`（最新的） | `{columns, data(前100行), stats{total, attack_count, normal_count, attack_ratio}}` |

---

## 四、API 接口详细说明

### 4.1 页面接口

#### `GET /`
- **功能**: 返回主页面
- **视图**: `views.index`
- **返回**: 渲染 `ids/index.html` 模板

---

### 4.2 任务触发接口（均为 POST，JSON Body）

#### `POST /api/preprocess/`
- **功能**: 启动数据预处理任务
- **请求体**:
  ```json
  {
    "dataset": "cicids2017",   // cicids2017 | nsl_kdd
    "data_dir": "./data"
  }
  ```
- **响应**:
  ```json
  {"success": true, "message": "数据预处理已开始"}
  // 或
  {"success": false, "message": "已有任务正在运行，请等待完成"}
  ```
- **后端行为**: 执行 `python data_preprocessing.py --dataset {dataset} --data_dir {data_dir} [--preprocess_only]`

---

#### `POST /api/train/`
- **功能**: 启动模型训练任务
- **请求体**:
  ```json
  {
    "dataset": "cicids2017",
    "model": "cnn_lstm",       // cnn | lstm | cnn_lstm
    "batch_size": 64,
    "epochs": 30,
    "lr": 0.001,
    "hidden_dim": 128,
    "num_layers": 2,
    "no_cuda": false
  }
  ```
- **响应**: 同上格式
- **后端行为**: 执行 `python train.py ...`
- **完成回调**: 在 `result_data` 中设置 `training_image` 路径

---

#### `POST /api/evaluate/`
- **功能**: 启动模型评估任务
- **请求体**:
  ```json
  {
    "dataset": "cicids2017",
    "model": "cnn",
    "batch_size": 64,
    "hidden_dim": 128,
    "num_layers": 2,
    "no_cuda": false,
    "data_dir": "./data"
  }
  ```
- **响应**: 同上格式
- **后端行为**: 执行 `python evaluate.py ...`，结果保存到 `results/{model}_{dataset}/`
- **完成回调**: 在 `result_data` 中设置 `results_dir` 路径

---

#### `POST /api/capture/`
- **功能**: 启动实时流量捕获任务
- **请求体**:
  ```json
  {
    "capture_time": 60,
    "data_dir": "./data"
  }
  ```
- **响应**: 同上格式
- **后端行为**: 执行 `python main.py --task capture --capture_time {time} --capture_file {file}`
- **完成回调**: 在 `result_data` 中设置 `capture_file` 路径
- **注意**: 需要管理员权限

---

#### `POST /api/detect/`
- **功能**: 启动威胁检测任务
- **请求体**:
  ```json
  {
    "dataset": "cicids2017",
    "model": "cnn",
    "no_cuda": false,
    "data_dir": "./data"
  }
  ```
- **响应**: 同上格式，如果捕获文件不存在返回 `{"success": false, "message": "捕获文件不存在: ..."}`
- **后端行为**: 执行 `python main.py --task detect ...`
- **前置条件**: 必须先执行过流量捕获，`data/capture_flows/captured_flows.csv` 存在
- **完成回调**: 在 `result_data` 中设置 `detection_file` 路径

---

### 4.3 状态与数据查询接口（GET）

#### `GET /api/task_status/?since=N`
- **功能**: 轮询当前任务状态和增量日志
- **参数**: `since` — 日志行偏移量（从第几行开始取新日志）
- **响应**:
  ```json
  {
    "is_running": true,
    "status": "processing",       // ready | processing | success | error
    "status_text": "模型训练中...",
    "task_name": "模型训练",
    "log_count": 42,
    "result_data": {},
    "new_logs": ["Epoch 1/30 ...", "..."],
    "log_offset": 42
  }
  ```
- **轮询机制**: 前端每 1 秒调用一次，当 `is_running` 为 `false` 时停止轮询

---

#### `GET /api/export_log/`
- **功能**: 导出当前任务的全部日志
- **返回**: `text/plain` 文件下载（`ids_log.txt`）

---

#### `GET /api/result_image/{image_name}/?dataset=xxx&model=xxx`
- **功能**: 获取评估结果图片
- **URL 参数**:
  - `image_name`: 图片文件名（`confusion_matrix.png` / `roc_curve.png` / `pr_curve.png` / `training_history.png`）
  - `dataset`: 数据集名
  - `model`: 模型名
- **图片查找顺序**:
  1. `{BASE_DIR}/{image_name}`
  2. `{BASE_DIR}/results/{model}_{dataset}/{image_name}`
  3. `{BASE_DIR}/results/{image_name}`
- **返回**: `image/png` 二进制流，或 404

---

#### `GET /api/capture_data/?data_dir=./data`
- **功能**: 获取捕获流量的表格数据
- **数据来源**: `{data_dir}/capture_flows/captured_flows.csv`
- **响应**:
  ```json
  {
    "columns": ["Flow Duration", "Total Fwd Packets", ...],
    "data": [[...], [...]],   // 前 100 行
    "total": 500
  }
  ```

---

#### `GET /api/detection_result/`
- **功能**: 获取最新的检测结果表格数据
- **数据来源**: `results/` 目录下最新的 `detection_result_*.csv`
- **响应**:
  ```json
  {
    "columns": ["prediction", "is_attack"],
    "data": [[0, false], [1, true], ...],   // 前 100 行
    "stats": {
      "total": 500,
      "attack_count": 23,
      "normal_count": 477,
      "attack_ratio": 4.6
    }
  }
  ```

---

## 五、前端轮询与任务完成数据流时序图

```
前端                          Django (views.py)              TaskManager              子进程
 │                                │                            │                       │
 │── POST /api/train/ ──────────→│                            │                       │
 │                                │── start_task(cmd) ────────→│                       │
 │                                │                            │── Popen(cmd) ─────────→│
 │←── {success:true} ────────────│                            │                       │
 │                                │                            │                       │
 │ [每1秒轮询]                     │                            │                       │
 │── GET /api/task_status/ ──────→│                            │                       │
 │                                │── get_status() ───────────→│                       │
 │                                │── get_logs(since) ─────────→│  ←─ stdout 逐行 ──── │
 │←── {is_running:true,           │                            │                       │
 │     new_logs:[...]} ──────────│                            │                       │
 │                                │                            │                       │
 │   ... 重复轮询 ...             │                            │                       │
 │                                │                            │                       │
 │                                │                            │  ←─ process.wait() ── │
 │                                │                            │── on_complete() ──→ result_data
 │── GET /api/task_status/ ──────→│                            │                       │
 │←── {is_running:false,          │                            │                       │
 │     result_data:{...}} ───────│                            │                       │
 │                                │                            │                       │
 │ [根据 result_data 类型]         │                            │                       │
 │── GET /api/result_image/ ─────→│ (读取 PNG 文件返回)         │                       │
 │── GET /api/capture_data/ ─────→│ (读取 CSV 返回 JSON)       │                       │
 │── GET /api/detection_result/ ─→│ (读取 CSV + 统计返回 JSON) │                       │
```

---

## 六、关键文件产出物汇总

| 产出物 | 路径 | 由谁生成 | 由谁消费 |
|-------|------|---------|---------|
| 预处理数据 | `data/{dataset}_processed/*.npy` | `data_preprocessing.py` | `train.py`, `evaluate.py`（通过 `data_utils`） |
| 训练模型权重 | `saved_models/{model}_{dataset}_model.pth` | `train.py` | `evaluate.py`, `main.py --task detect` |
| 训练历史图 | `training_history.png` | `train.py` → `training.plot_training_history()` | 前端可视化 Tab |
| 混淆矩阵图 | `results/{model}_{dataset}/confusion_matrix.png` | `evaluate.py` | 前端可视化 Tab |
| ROC 曲线图 | `results/{model}_{dataset}/roc_curve.png` | `evaluate.py` | 前端可视化 Tab |
| PR 曲线图 | `results/{model}_{dataset}/pr_curve.png` | `evaluate.py` | 前端可视化 Tab |
| 捕获流量 CSV | `data/capture_flows/captured_flows.csv` | `capture_to_csv.py` | 前端流量数据 Tab, `main.py --task detect` |
| 检测结果 CSV | `results/detection_result_{timestamp}.csv` | `main.py --task detect` | 前端检测结果 Tab |
| 检测结果图 | `results/detection_result_{timestamp}.png` | `main.py --task detect` | （未在前端展示） |
