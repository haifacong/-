import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from scipy import sparse
import warnings
warnings.filterwarnings("ignore")

def load_nsl_kdd(data_path, test_size=0.2, random_state=42):
    # 读取数据
    columns = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", 
              "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", 
              "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations", 
              "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login", 
              "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate", 
              "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", 
              "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate", 
              "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", 
              "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate", 
              "dst_host_srv_rerror_rate", "class", "difficulty"]
    
    # 读取训练集
    train_path = os.path.join(data_path, "KDDTrain+.txt")
    train_df = pd.read_csv(train_path, header=None, names=columns)
    
    # 读取测试集
    test_path = os.path.join(data_path, "KDDTest+.txt")
    test_df = pd.read_csv(test_path, header=None, names=columns)
    
    # 合并数据集以进行预处理
    df = pd.concat([train_df, test_df], axis=0)
    
    # 删除difficulty列（不需要）
    df.drop(['difficulty'], axis=1, inplace=True)
    
    # 处理标签
    df['binary_class'] = df['class'].apply(lambda x: 0 if x == 'normal' else 1)
    
    # 分离特征和标签
    X = df.drop(['class', 'binary_class'], axis=1)
    y = df['binary_class']
    
    # 识别数值和分类特征
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    
    # 创建预处理管道
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    # 拟合和转换数据
    X_processed = preprocessor.fit_transform(X)
    
    # 将处理后的数据重新分割为训练集和测试集
    X_train_processed, X_test_processed, y_train, y_test = train_test_split(
        X_processed, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train_processed, X_test_processed, y_train.values, y_test.values, preprocessor

def load_unsw_nb15(data_path, test_size=0.2, random_state=42):

    train_path = os.path.join(data_path, "UNSW_NB15_training-set.csv")
    test_path = os.path.join(data_path, "UNSW_NB15_testing-set.csv")
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    df = pd.concat([train_df, test_df], axis=0)
    
    # 删除id和label列（我们使用attack_cat作为标签）
    df.drop(['id'], axis=1, inplace=True)
    
    # 处理标签 - 二分类问题
    # label: 0 表示正常，1 表示攻击
    y = df['label']
    X = df.drop(['label', 'attack_cat'], axis=1)
    
    # 识别数值和分类特征
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    
    # 创建预处理管道
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    # 拟合和转换数据
    X_processed = preprocessor.fit_transform(X)
    
    # 将处理后的数据重新分割为训练集和测试集
    X_train_processed, X_test_processed, y_train, y_test = train_test_split(
        X_processed, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train_processed, X_test_processed, y_train.values, y_test.values, preprocessor

def load_kdd_cup99(data_path, test_size=0.2, random_state=42, use_10_percent=True):
    columns = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", 
              "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", 
              "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations", 
              "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login", 
              "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate", 
              "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", 
              "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate", 
              "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", 
              "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate", 
              "dst_host_srv_rerror_rate", "class"]
    
    # 读取数据
    if use_10_percent:
        # 使用10%版本的数据集
        file_path = os.path.join(data_path, "kddcup.data_10_percent.gz")
        if not os.path.exists(file_path):
            file_path = os.path.join(data_path, "kddcup.data_10_percent")
    else:
        # 使用完整版数据集
        file_path = os.path.join(data_path, "kddcup.data.gz")
        if not os.path.exists(file_path):
            file_path = os.path.join(data_path, "kddcup.data")
    
    print(f"加载KDD Cup 99数据集：{file_path}")
    df = pd.read_csv(file_path, header=None, names=columns)
    print(f"数据集大小：{len(df)}行")
    
    # 处理标签 - 二分类问题（正常与攻击）
    df['binary_class'] = df['class'].apply(lambda x: 0 if x == 'normal.' else 1)
    
    # 打印各类别数量
    print("类别分布:")
    print(df['binary_class'].value_counts())
    
    # 分离特征和标签
    X = df.drop(['class', 'binary_class'], axis=1)
    y = df['binary_class']
    
    # 识别数值和分类特征
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    
    # 创建预处理管道
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    # 拟合和转换数据
    X_processed = preprocessor.fit_transform(X)
    
    # 将处理后的数据重新分割为训练集和测试集
    X_train_processed, X_test_processed, y_train, y_test = train_test_split(
        X_processed, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train_processed, X_test_processed, y_train.values, y_test.values, preprocessor

def load_cicids2017(data_path, test_size=0.2, random_state=42):
    meaningful_features = [
        'Flow Duration',
        'Total Fwd Packets',
        'Total Backward Packets',
        'Total Length of Fwd Packets',
        'Total Length of Bwd Packets',
        'Fwd Packet Length Max',
        'Fwd Packet Length Min',
        'Fwd Packet Length Mean',
        'Bwd Packet Length Max',
        'Bwd Packet Length Min',
        'Bwd Packet Length Mean',
        'Flow Bytes/s',
        'Flow Packets/s',
        'Fwd Header Length',
        'Bwd Header Length',
        'Fwd Packets/s',
        'Bwd Packets/s',
        'Min Packet Length',
        'Max Packet Length',
        'Packet Length Mean',
        'Packet Length Std',
        'Packet Length Variance',
        'Fwd IAT Mean',
        'Bwd IAT Mean',
        'Active Mean',
        'Idle Mean',
    ]
    
    all_files = []
    for file in os.listdir(data_path):
        if file.endswith('.csv'):
            file_path = os.path.join(data_path, file)
            df = pd.read_csv(file_path)
            all_files.append(df)
    
    df = pd.concat(all_files, axis=0, ignore_index=True)
    X = df[meaningful_features]
    y = df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
    
    # 处理无穷大和异常值
    # 1. 将无穷大替换为NaN
    X = X.replace([np.inf, -np.inf], np.nan)
    
    # 2. 使用中位数填充NaN值
    X = X.fillna(X.median())
    
    # 3. 处理异常值（使用IQR方法）
    for column in X.columns:
        Q1 = X[column].quantile(0.25)
        Q3 = X[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # 将超出范围的值替换为边界值
        X[column] = X[column].clip(lower_bound, upper_bound)
    
    # 创建预处理管道
    preprocessor = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    # 拟合和转换数据
    X_processed = preprocessor.fit_transform(X)
    
    # 将处理后的数据分割为训练集和测试集
    X_train_processed, X_test_processed, y_train, y_test = train_test_split(
        X_processed, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train_processed, X_test_processed, y_train.values, y_test.values, preprocessor

def load_captured_traffic(capture_file):
    print(f"加载捕获的流量数据: {capture_file}")
    df = pd.read_csv(capture_file)
    print(f"加载了 {len(df)} 条流量记录")
    
    meaningful_features = [
        'Flow Duration',
        'Total Fwd Packets',
        'Total Backward Packets',
        'Total Length of Fwd Packets',
        'Total Length of Bwd Packets',
        'Fwd Packet Length Max',
        'Fwd Packet Length Min',
        'Fwd Packet Length Mean',
        'Bwd Packet Length Max',
        'Bwd Packet Length Min',
        'Bwd Packet Length Mean',
        'Flow Bytes/s',
        'Flow Packets/s',
        'Fwd Header Length',
        'Bwd Header Length',
        'Fwd Packets/s',
        'Bwd Packets/s',
        'Min Packet Length',
        'Max Packet Length',
        'Packet Length Mean',
        'Packet Length Std',
        'Packet Length Variance',
        'Fwd IAT Mean',
        'Bwd IAT Mean',
        'Active Mean',
        'Idle Mean',
    ]
    
    missing_features = [f for f in meaningful_features if f not in df.columns]
    if missing_features:
        print(f"警告: 捕获的数据缺少以下特征: {missing_features}")
        meaningful_features = [f for f in meaningful_features if f in df.columns]
        print(f"将使用以下特征: {meaningful_features}")
    
    # 选择特征
    X = df[meaningful_features]
    
    # 处理无穷大和异常值
    # 1. 将无穷大替换为NaN
    X = X.replace([np.inf, -np.inf], np.nan)
    
    # 2. 使用中位数填充NaN值
    X = X.fillna(X.median())
    
    # 3. 处理异常值（使用IQR方法）
    for column in X.columns:
        Q1 = X[column].quantile(0.25)
        Q3 = X[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # 将超出范围的值替换为边界值
        X[column] = X[column].clip(lower_bound, upper_bound)
    
    # 创建预处理管道
    preprocessor = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    # 拟合和转换数据
    X_processed = preprocessor.fit_transform(X)
    
    return X_processed, preprocessor

def get_dataset_loader(dataset_name, data_path, batch_size=64, test_size=0.2, random_state=42, use_10_percent=True):
    import torch
    from torch.utils.data import TensorDataset, DataLoader

    if dataset_name.lower() == 'nsl_kdd':
        X_train, X_test, y_train, y_test, _ = load_nsl_kdd(data_path, test_size, random_state)
    elif dataset_name.lower() == 'unsw_nb15':
        X_train, X_test, y_train, y_test, _ = load_unsw_nb15(data_path, test_size, random_state)
    elif dataset_name.lower() == 'kdd_cup99':
        X_train, X_test, y_train, y_test, _ = load_kdd_cup99(data_path, test_size, random_state, use_10_percent)
    elif dataset_name.lower() == 'cicids2017':
        X_train, X_test, y_train, y_test, _ = load_cicids2017(data_path, test_size, random_state)
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")
    
    if sparse.issparse(X_train):
        X_train = X_train.toarray()
    if sparse.issparse(X_test):
        X_test = X_test.toarray()
    
    # 转换为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_train_tensor = torch.LongTensor(y_train)
    y_test_tensor = torch.LongTensor(y_test)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    feature_dim = X_train.shape[1]

    return train_loader, test_loader, feature_dim 