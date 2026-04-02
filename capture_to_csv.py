import os
import time
import pandas as pd
import numpy as np
import argparse
import threading
import warnings
from scapy.sendrecv import sniff
warnings.filterwarnings("ignore")
from collections import defaultdict
from scapy.layers.inet import IP, TCP, UDP
from datetime import datetime

MEANINGFUL_FEATURES = [
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
    'Min Idle',
    'Max Idle'
]

FLOW_TIMEOUT = 120  # 流超时时间(秒)

# 流类，用于跟踪和收集网络流的统计信息
class Flow:
    def __init__(self, src_ip, dst_ip, src_port, dst_port, protocol):
        self.src_ip = src_ip
        self.dst_ip = dst_ip
        self.src_port = src_port
        self.dst_port = dst_port
        self.protocol = protocol
        self.forward_packets = []  # 存储正向数据包
        self.backward_packets = []  # 存储反向数据包
        self.start_time = time.time()
        self.last_seen = time.time()
        
        # 统计变量
        self.fwd_packet_count = 0
        self.bwd_packet_count = 0
        self.fwd_byte_count = 0
        self.bwd_byte_count = 0
        self.fwd_total_len = 0
        self.bwd_total_len = 0
        
        # 时间统计
        self.flow_duration = 0
        self.fwd_header_len = 0
        self.bwd_header_len = 0
        self.fwd_iat_total = 0  # 前向包间到达时间总和
        self.bwd_iat_total = 0  # 后向包间到达时间总和
        self.active_time = 0
        self.idle_time = 0
        
        # 标志和计数器
        self.is_complete = False
    
    def add_packet(self, packet, direction):
        current_time = time.time()
        
        if direction == 'forward':
            if self.fwd_packet_count > 0:
                self.fwd_iat_total += current_time - self.forward_packets[-1]['time']
            
            self.fwd_packet_count += 1
            packet_len = len(packet)
            self.fwd_byte_count += packet_len
            self.fwd_total_len += packet_len
            
            header_len = 20  
            if packet.haslayer(TCP):
                header_len += 20  
            elif packet.haslayer(UDP):
                header_len += 8  
            
            self.fwd_header_len += header_len
            
            # 存储数据包信息
            self.forward_packets.append({
                'time': current_time,
                'size': packet_len,
                'header_len': header_len
            })
            
        elif direction == 'backward':
            # 添加后向数据包
            if self.bwd_packet_count > 0:
                self.bwd_iat_total += current_time - self.backward_packets[-1]['time']
            
            self.bwd_packet_count += 1
            packet_len = len(packet)
            self.bwd_byte_count += packet_len
            self.bwd_total_len += packet_len
            
            header_len = 20  
            if packet.haslayer(TCP):
                header_len += 20 
            elif packet.haslayer(UDP):
                header_len += 8 
            
            self.bwd_header_len += header_len
            
            # 存储数据包信息
            self.backward_packets.append({
                'time': current_time,
                'size': packet_len,
                'header_len': header_len
            })
        
        # 更新流持续时间和最后一次看到的时间
        self.flow_duration = current_time - self.start_time
        self.last_seen = current_time
        
        if self.flow_duration > FLOW_TIMEOUT:
            self.is_complete = True
    
    def is_expired(self, current_time=None):
        if current_time is None:
            current_time = time.time()
        return (current_time - self.last_seen) > FLOW_TIMEOUT
    
    def extract_features(self):
        if self.fwd_packet_count == 0 and self.bwd_packet_count == 0:
            return None 
        
        # 基本特征
        features = {
            'Flow Duration': self.flow_duration * 1000, 
            'Total Fwd Packets': self.fwd_packet_count,
            'Total Backward Packets': self.bwd_packet_count,
            'Total Length of Fwd Packets': self.fwd_total_len,
            'Total Length of Bwd Packets': self.bwd_total_len,
            'Fwd Packet Length Max': max([p['size'] for p in self.forward_packets]) if self.forward_packets else 0,
            'Fwd Packet Length Min': min([p['size'] for p in self.forward_packets]) if self.forward_packets else 0,
            'Fwd Packet Length Mean': self.fwd_total_len / self.fwd_packet_count if self.fwd_packet_count > 0 else 0,
            'Bwd Packet Length Max': max([p['size'] for p in self.backward_packets]) if self.backward_packets else 0,
            'Bwd Packet Length Min': min([p['size'] for p in self.backward_packets]) if self.backward_packets else 0,
            'Bwd Packet Length Mean': self.bwd_total_len / self.bwd_packet_count if self.bwd_packet_count > 0 else 0,
            'Flow Bytes/s': (self.fwd_total_len + self.bwd_total_len) / self.flow_duration if self.flow_duration > 0 else 0,
            'Flow Packets/s': (self.fwd_packet_count + self.bwd_packet_count) / self.flow_duration if self.flow_duration > 0 else 0,
            'Fwd Header Length': self.fwd_header_len,
            'Bwd Header Length': self.bwd_header_len,
            'Fwd Packets/s': self.fwd_packet_count / self.flow_duration if self.flow_duration > 0 else 0,
            'Bwd Packets/s': self.bwd_packet_count / self.flow_duration if self.flow_duration > 0 else 0,
            'Min Packet Length': min([p['size'] for p in self.forward_packets + self.backward_packets]) if (self.forward_packets + self.backward_packets) else 0,
            'Max Packet Length': max([p['size'] for p in self.forward_packets + self.backward_packets]) if (self.forward_packets + self.backward_packets) else 0,
            'Packet Length Mean': (self.fwd_total_len + self.bwd_total_len) / (self.fwd_packet_count + self.bwd_packet_count) if (self.fwd_packet_count + self.bwd_packet_count) > 0 else 0,
            'Packet Length Std': np.std([p['size'] for p in self.forward_packets + self.backward_packets]) if (self.forward_packets + self.backward_packets) else 0,
            'Packet Length Variance': np.var([p['size'] for p in self.forward_packets + self.backward_packets]) if (self.forward_packets + self.backward_packets) else 0,
            'Fwd IAT Mean': self.fwd_iat_total / (self.fwd_packet_count - 1) if self.fwd_packet_count > 1 else 0,
            'Bwd IAT Mean': self.bwd_iat_total / (self.bwd_packet_count - 1) if self.bwd_packet_count > 1 else 0,
            'Active Mean': self.active_time / (self.fwd_packet_count + self.bwd_packet_count) if (self.fwd_packet_count + self.bwd_packet_count) > 0 else 0,
            'Idle Mean': self.idle_time / (self.fwd_packet_count + self.bwd_packet_count) if (self.fwd_packet_count + self.bwd_packet_count) > 0 else 0,
            'Min Idle': 0,
            'Max Idle': 0
        }
        
        for feature in MEANINGFUL_FEATURES:
            if feature not in features:
                features[feature] = 0
        
        features = {k: features[k] for k in MEANINGFUL_FEATURES}
        
        return features
    
    def get_flow_info(self):
        return {
            'src_ip': self.src_ip,
            'dst_ip': self.dst_ip,
            'src_port': self.src_port,
            'dst_port': self.dst_port,
            'protocol': self.protocol,
            'duration': self.flow_duration,
            'fwd_packets': self.fwd_packet_count,
            'bwd_packets': self.bwd_packet_count,
            'total_bytes': self.fwd_total_len + self.bwd_total_len
        }

# 流管理器，用于跟踪和管理网络流
class FlowManager:
    def __init__(self):
        self.flows = {}  # 使用 5 元组作为键：(src_ip, dst_ip, src_port, dst_port, protocol)
        self.completed_flows = []
        self.lock = threading.Lock()
    
    def process_packet(self, packet):
        with self.lock:
            if not packet.haslayer(IP):
                return
            
            ip_packet = packet[IP]
            src_ip = ip_packet.src
            dst_ip = ip_packet.dst
            
            # 获取协议和端口
            if ip_packet.haslayer(TCP):
                protocol = 'TCP'
                transport_packet = ip_packet[TCP]
                src_port = transport_packet.sport
                dst_port = transport_packet.dport
            elif ip_packet.haslayer(UDP):
                protocol = 'UDP'
                transport_packet = ip_packet[UDP]
                src_port = transport_packet.sport
                dst_port = transport_packet.dport
            else:
                return
            
            forward_key = (src_ip, dst_ip, src_port, dst_port, protocol)
            backward_key = (dst_ip, src_ip, dst_port, src_port, protocol)
            
            if forward_key in self.flows:
                self.flows[forward_key].add_packet(packet, 'forward')
                flow = self.flows[forward_key]
            elif backward_key in self.flows:
                self.flows[backward_key].add_packet(packet, 'backward')
                flow = self.flows[backward_key]
            else:
                flow = Flow(src_ip, dst_ip, src_port, dst_port, protocol)
                flow.add_packet(packet, 'forward')
                self.flows[forward_key] = flow
            
            if flow.is_complete:
                self.completed_flows.append(flow)
                if forward_key in self.flows:
                    del self.flows[forward_key]
                elif backward_key in self.flows:
                    del self.flows[backward_key]
    
    def clean_expired_flows(self):
        with self.lock:
            current_time = time.time()
            expired_keys = []
            
            for key, flow in self.flows.items():
                if flow.is_expired(current_time):
                    self.completed_flows.append(flow)
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.flows[key]

def capture_and_save_to_csv(duration=60, output_path=None, debug=True):
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = "captured_data"
        os.makedirs(output_dir, exist_ok=True)
        output_path = f"{output_dir}/captured_flows_{timestamp}.csv"
    else:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
    
    flow_manager = FlowManager()
    
    # 启动清理线程，定期清理过期流
    def clean_flows_periodically(interval=10):
        while True:
            time.sleep(interval)
            flow_manager.clean_expired_flows()
    
    cleaning_thread = threading.Thread(target=clean_flows_periodically)
    cleaning_thread.daemon = True
    cleaning_thread.start()
    
    # 显示开始信息
    print(f"开始捕获网络流量，持续 {duration} 秒...")
    print(f"捕获时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"输出文件: {output_path}")
    
    end_time = time.time() + duration
    packet_count = 0
    
    # 数据包回调函数
    def packet_callback(packet):
        nonlocal packet_count
        flow_manager.process_packet(packet)
        packet_count += 1
        if packet_count % 100 == 0 and debug:
            print(f"已捕获 {packet_count} 个数据包...")
        return time.time() < end_time
    
    # 捕获数据包
    try:
        sniff(prn=packet_callback, store=False, timeout=duration)
        print("捕获结束")
    except Exception as e:
        print(f"捕获过程中发生错误: {e}")
        if packet_count == 0:
            print("未捕获到任何数据包")
            return None
    
    # 获取所有流
    with flow_manager.lock:
        flows = list(flow_manager.flows.values())
        completed_flows = flow_manager.completed_flows
    
    all_flows = flows + completed_flows
    total_flows = len(all_flows)
    
    print(f"\n总共捕获了 {total_flows} 个流, {packet_count} 个数据包")
    
    if total_flows == 0:
        print("没有捕获到任何流量，无法保存CSV文件")
        return None
    
    # 提取特征
    all_features = []
    valid_flow_count = 0
    
    for i, flow in enumerate(all_flows):
        features = flow.extract_features()
        if features is None:
            continue
        
        valid_flow_count += 1
        all_features.append(features)
        
        # 输出每个流的基本信息（如果开启调试）
        if debug:
            flow_info = flow.get_flow_info()
            if i < 10 or i % 50 == 0:  # 只显示前10个和之后每50个的信息
                print(f"\n流 {i+1}/{total_flows}: {flow_info['src_ip']}:{flow_info['src_port']} -> {flow_info['dst_ip']}:{flow_info['dst_port']} ({flow_info['protocol']})")
                print(f"持续时间: {flow_info['duration']:.2f}秒, 前向包: {flow_info['fwd_packets']}, 后向包: {flow_info['bwd_packets']}, 总字节: {flow_info['total_bytes']}")
    
    if valid_flow_count == 0:
        print("没有有效的流量特征，无法保存CSV文件")
        return None
    
    # 创建数据框并保存为CSV
    df = pd.DataFrame(all_features)
    
    for feature in MEANINGFUL_FEATURES:
        if feature not in df.columns:
            df[feature] = 0
    df = df[MEANINGFUL_FEATURES]
    df.to_csv(output_path, index=False)
    print(f"\n成功保存 {valid_flow_count} 条流记录到 {output_path}")
    
    # 显示特征统计信息
    if debug:
        print(f"\n特征维度: {df.shape}")
        print(f"特征数量: {len(df.columns)}")
        print("\n特征值统计:")
        stats = df.describe().transpose()
        stats_to_show = ['count', 'mean', 'min', 'max']
        columns_exist = [col for col in stats_to_show if col in stats.columns]
        print(stats[columns_exist].head(10))  # 只显示前10个特征的统计信息
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description="捕获网络流量并保存为CSV文件")
    parser.add_argument("-t", "--time", type=int, default=60, help="捕获时间(秒)，默认为60秒")
    parser.add_argument("-o", "--output", type=str, help="输出CSV文件路径")
    parser.add_argument("-d", "--debug", action="store_true", help="显示调试信息")
    args = parser.parse_args()
    
    try:
        output_file = capture_and_save_to_csv(
            duration=args.time,
            output_path=args.output,
            debug=args.debug
        )
        
        if output_file:
            print(f"\n捕获完成! 数据已保存到: {output_file}")
        else:
            print("\n捕获失败或未捕获到有效流量")
    
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"\n程序发生错误: {e}")

if __name__ == "__main__":
    main() 