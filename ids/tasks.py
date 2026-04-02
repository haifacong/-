import subprocess
import threading
import time
import os


class TaskManager:
    """全局任务管理器，替代 PyQt5 的 Worker(QThread)"""

    def __init__(self):
        self.current_task = None
        self.task_name = None
        self.log_lines = []
        self.is_running = False
        self.status = 'ready'  # ready / processing / success / error
        self.status_text = '就绪'
        self._lock = threading.Lock()
        self._thread = None

        # 任务完成后的结果
        self.result_data = {}

    def start_task(self, task_name, command, on_complete=None):
        """启动后台任务"""
        with self._lock:
            if self.is_running:
                return False, '已有任务正在运行，请等待完成'

            self.log_lines = []
            self.task_name = task_name
            self.is_running = True
            self.status = 'processing'
            self.status_text = f'{task_name}中...'
            self.result_data = {}

        self._thread = threading.Thread(
            target=self._run_command,
            args=(command, on_complete),
            daemon=True
        )
        self._thread.start()
        return True, f'{task_name}已开始'

    def _run_command(self, command, on_complete=None):
        """在子线程中执行命令"""
        try:
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                universal_newlines=True,
                encoding='UTF-8',
                errors='replace',
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )

            for line in process.stdout:
                stripped = line.strip()
                if stripped and stripped not in ('True', 'False'):
                    with self._lock:
                        self.log_lines.append(stripped)

            process.wait()

            with self._lock:
                if process.returncode == 0:
                    self.status = 'success'
                    self.status_text = f'{self.task_name}完成'
                else:
                    self.status = 'error'
                    self.status_text = f'{self.task_name}失败'

        except Exception as e:
            with self._lock:
                self.log_lines.append(f'[ERROR] 执行命令时出错: {str(e)}')
                self.status = 'error'
                self.status_text = f'{self.task_name}失败'
        finally:
            with self._lock:
                self.is_running = False
            if on_complete:
                try:
                    on_complete()
                except Exception:
                    pass

    def get_logs(self, since=0):
        """获取从 since 行开始的新日志"""
        with self._lock:
            return self.log_lines[since:]

    def get_status(self):
        """获取当前状态"""
        with self._lock:
            return {
                'is_running': self.is_running,
                'status': self.status,
                'status_text': self.status_text,
                'task_name': self.task_name,
                'log_count': len(self.log_lines),
                'result_data': self.result_data,
            }

    def clear_logs(self):
        """清空日志"""
        with self._lock:
            self.log_lines = []
            self.status = 'ready'
            self.status_text = '就绪'
            self.result_data = {}


# 全局单例
task_manager = TaskManager()
