// IDS Web 前端交互逻辑

const IDS = {
    pollTimer: null,
    logOffset: 0,
    dataDir: './data',

    // 获取当前表单参数
    getParams() {
        return {
            dataset: document.getElementById('dataset').value,
            model: document.getElementById('model').value,
            batch_size: parseInt(document.getElementById('batch_size').value),
            epochs: parseInt(document.getElementById('epochs').value),
            lr: parseFloat(document.getElementById('lr').value),
            hidden_dim: parseInt(document.getElementById('hidden_dim').value),
            num_layers: parseInt(document.getElementById('num_layers').value),
            no_cuda: document.getElementById('no_cuda').checked,
            data_dir: this.dataDir,
            capture_time: parseInt(document.getElementById('capture_time').value),
            detect_model: document.getElementById('detect_model').value,
        };
    },

    // 发起任务请求
    async postTask(url, params) {
        try {
            const resp = await fetch(url, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(params),
            });
            const data = await resp.json();
            if (!data.success) {
                this.appendLog('[错误] ' + data.message);
                return false;
            }
            this.logOffset = 0;
            this.startPolling();
            return true;
        } catch (e) {
            this.appendLog('[错误] 请求失败: ' + e.message);
            return false;
        }
    },

    // 开始轮询任务状态
    startPolling() {
        this.stopPolling();
        this.setProgress(true);
        this.pollTimer = setInterval(() => this.pollStatus(), 1000);
    },

    stopPolling() {
        if (this.pollTimer) {
            clearInterval(this.pollTimer);
            this.pollTimer = null;
        }
    },

    // 轮询任务状态
    async pollStatus() {
        try {
            const resp = await fetch('/api/task_status/?since=' + this.logOffset);
            const data = await resp.json();

            // 追加新日志
            if (data.new_logs && data.new_logs.length > 0) {
                data.new_logs.forEach(line => this.appendLog(line));
                this.logOffset = data.log_offset;
            }

            // 更新状态栏
            this.setStatus(data.status, data.status_text);

            // 任务完成，停止轮询
            if (!data.is_running) {
                this.stopPolling();
                this.setProgress(false);
                this.onTaskComplete(data);
            }
        } catch (e) {
            // 静默处理轮询错误
        }
    },

    // 任务完成后的处理
    onTaskComplete(data) {
        const result = data.result_data || {};

        if (result.training_image) {
            this.showResultImage('training_history.png');
            this.switchTab('tab-visual');
        }

        if (result.results_dir) {
            this.showResultImage('confusion_matrix.png');
            this.switchTab('tab-visual');
        }

        if (result.capture_file) {
            this.loadCaptureData();
            this.switchTab('tab-capture');
        }

        if (result.detection_file) {
            this.loadDetectionResult();
            this.switchTab('tab-detect');
        }
    },

    // === UI 操作 ===

    appendLog(text) {
        const box = document.getElementById('output-box');
        box.textContent += text + '\n';
        box.scrollTop = box.scrollHeight;
    },

    clearOutput() {
        document.getElementById('output-box').textContent = '';
        document.getElementById('image-display').innerHTML = '<span class="placeholder">图像将在这里显示</span>';
        this.logOffset = 0;
        this.setStatus('ready', '就绪');
        fetch('/api/task_status/');  // reset server logs
    },

    setStatus(status, text) {
        const bar = document.getElementById('status-bar');
        bar.className = 'status-bar';
        if (status === 'processing') {
            bar.classList.add('status-processing');
            bar.textContent = text + ' ...';
        } else if (status === 'error') {
            bar.classList.add('status-error');
            bar.textContent = text + ' ✗';
        } else {
            bar.classList.add('status-ready');
            bar.textContent = (text || '就绪') + ' ✓';
        }
    },

    setProgress(active) {
        const bar = document.getElementById('progress-bar');
        if (active) {
            bar.classList.add('active');
        } else {
            bar.classList.remove('active');
        }
    },

    setButtonsDisabled(disabled) {
        document.querySelectorAll('.btn-primary').forEach(btn => {
            btn.disabled = disabled;
        });
    },

    // === Tab 切换 ===

    switchTab(tabId) {
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.tab === tabId);
        });
        document.querySelectorAll('.tab-pane').forEach(pane => {
            pane.classList.toggle('active', pane.id === tabId);
        });
    },

    // === 任务触发 ===

    preprocess() {
        this.clearOutput();
        const p = this.getParams();
        this.appendLog('开始数据预处理...');
        this.appendLog('数据集: ' + p.dataset);
        this.postTask('/api/preprocess/', { dataset: p.dataset, data_dir: p.data_dir });
    },

    train() {
        this.clearOutput();
        const p = this.getParams();
        this.appendLog('开始训练 ' + p.model.toUpperCase() + ' 模型...');
        this.appendLog('数据集: ' + p.dataset + ' | batch_size=' + p.batch_size + ' | epochs=' + p.epochs + ' | lr=' + p.lr);
        this.postTask('/api/train/', {
            dataset: p.dataset, model: p.model,
            batch_size: p.batch_size, epochs: p.epochs, lr: p.lr,
            hidden_dim: p.hidden_dim, num_layers: p.num_layers, no_cuda: p.no_cuda,
        });
    },

    evaluate() {
        this.clearOutput();
        const p = this.getParams();
        this.appendLog('开始评估 ' + p.model.toUpperCase() + ' 模型...');
        this.postTask('/api/evaluate/', {
            dataset: p.dataset, model: p.model,
            batch_size: p.batch_size, hidden_dim: p.hidden_dim,
            num_layers: p.num_layers, no_cuda: p.no_cuda, data_dir: p.data_dir,
        });
    },

    capture() {
        this.clearOutput();
        const p = this.getParams();
        this.appendLog('开始捕获网络流量，持续 ' + p.capture_time + ' 秒...');
        this.postTask('/api/capture/', { capture_time: p.capture_time, data_dir: p.data_dir });
    },

    detect() {
        this.clearOutput();
        const p = this.getParams();
        this.appendLog('开始威胁检测...');
        this.appendLog('检测模型: ' + p.detect_model.toUpperCase() + ' | 数据集: ' + p.dataset);
        this.postTask('/api/detect/', {
            dataset: p.dataset, model: p.detect_model,
            no_cuda: p.no_cuda, data_dir: p.data_dir,
        });
    },

    exportLog() {
        window.open('/api/export_log/', '_blank');
    },

    // === 数据加载 ===

    showResultImage(imageName) {
        const p = this.getParams();
        const url = '/api/result_image/' + imageName + '/?dataset=' + p.dataset + '&model=' + p.model + '&t=' + Date.now();
        const container = document.getElementById('image-display');
        container.innerHTML = '<img src="' + url + '" alt="' + imageName + '" onerror="this.parentElement.innerHTML=\'<span class=placeholder>图片未找到: ' + imageName + '</span>\'">';
    },

    async loadCaptureData() {
        try {
            const resp = await fetch('/api/capture_data/?data_dir=' + encodeURIComponent(this.dataDir));
            if (!resp.ok) return;
            const data = await resp.json();
            this.renderTable('capture-table', data.columns, data.data);
            this.appendLog('捕获流量统计: 总流量 ' + data.total + ' 条');
        } catch (e) {
            // ignore
        }
    },

    async loadDetectionResult() {
        try {
            const resp = await fetch('/api/detection_result/');
            if (!resp.ok) return;
            const data = await resp.json();
            this.renderTable('detect-table', data.columns, data.data);
            if (data.stats) {
                const s = data.stats;
                let info = '检测结果统计: 总流量 ' + s.total + ' 条';
                if (s.normal_count !== undefined) {
                    info += ' | 正常: ' + s.normal_count + ' | 攻击: ' + s.attack_count + ' (' + s.attack_ratio + '%)';
                }
                this.appendLog(info);
                this.renderStats('detect-stats', s);
            }
        } catch (e) {
            // ignore
        }
    },

    renderTable(containerId, columns, rows) {
        const container = document.getElementById(containerId);
        if (!columns || !rows || rows.length === 0) {
            container.innerHTML = '<p style="color: var(--color-text-muted)">暂无数据</p>';
            return;
        }
        let html = '<div class="table-wrapper"><table class="data-table"><thead><tr>';
        columns.forEach(col => { html += '<th>' + this.escapeHtml(String(col)) + '</th>'; });
        html += '</tr></thead><tbody>';
        rows.forEach(row => {
            html += '<tr>';
            row.forEach(cell => { html += '<td>' + this.escapeHtml(String(cell)) + '</td>'; });
            html += '</tr>';
        });
        html += '</tbody></table></div>';
        container.innerHTML = html;
    },

    renderStats(containerId, stats) {
        const container = document.getElementById(containerId);
        if (!stats || stats.normal_count === undefined) {
            container.innerHTML = '';
            return;
        }
        container.innerHTML = `
            <div class="stats-box">
                <span class="stat-item"><span class="stat-label">总流量: </span><span class="stat-value">${stats.total}</span></span>
                <span class="stat-item"><span class="stat-label">正常: </span><span class="stat-value stat-normal">${stats.normal_count}</span></span>
                <span class="stat-item"><span class="stat-label">攻击: </span><span class="stat-value stat-attack">${stats.attack_count}</span></span>
                <span class="stat-item"><span class="stat-label">攻击比例: </span><span class="stat-value stat-attack">${stats.attack_ratio}%</span></span>
            </div>`;
    },

    escapeHtml(str) {
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    },
};

// Tab 切换事件绑定
document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => IDS.switchTab(btn.dataset.tab));
    });
});
