"""
训练服务
管理训练进程的生命周期
参考 mikazuki/tasks.py
"""

import subprocess
import threading
import queue
import os
import sys
import psutil
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
from gui.backend.log import log


class TrainStatus(Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    COMPLETED = "completed"
    ERROR = "error"


class TrainerService:
    """训练服务单例"""
    
    def __init__(self):
        self.process: Optional[subprocess.Popen] = None
        self.status = TrainStatus.IDLE
        self.log_queue: queue.Queue = queue.Queue(maxsize=10000)
        self.logs: List[str] = []
        self.max_logs = 5000  # 保留最近5000行日志
        self.lock = threading.Lock()
        self.start_time: Optional[datetime] = None
        self.command: Optional[List[str]] = None
        self.env: Optional[Dict[str, str]] = None
        
        # 训练统计
        self.current_epoch = 0
        self.current_step = 0
        self.current_loss = 0.0
        
    def start_training(self, command: List[str], env: Dict[str, str]) -> Optional[str]:
        """启动训练进程"""
        with self.lock:
            if self.status == TrainStatus.RUNNING:
                log.warning("已有训练任务正在运行")
                return None
            
            # 清理之前的日志
            self.logs.clear()
            while not self.log_queue.empty():
                try:
                    self.log_queue.get_nowait()
                except queue.Empty:
                    break
            
            self.command = command
            self.env = env
            self.start_time = datetime.now()
            
            try:
                # 启动进程
                self.process = subprocess.Popen(
                    command,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    encoding='utf-8',
                    errors='replace'
                )
                
                self.status = TrainStatus.RUNNING
                
                # 启动日志读取线程
                log_thread = threading.Thread(target=self._read_logs, daemon=True)
                log_thread.start()
                
                # 启动监控线程
                monitor_thread = threading.Thread(target=self._monitor_process, daemon=True)
                monitor_thread.start()
                
                log.info(f"训练进程已启动，PID: {self.process.pid}")
                return str(self.process.pid)
                
            except Exception as e:
                log.error(f"启动训练进程失败: {e}")
                self.status = TrainStatus.ERROR
                return None
    
    def stop_training(self):
        """停止训练进程"""
        with self.lock:
            if self.process and self.process.poll() is None:
                try:
                    # 使用 psutil 终止进程树
                    parent = psutil.Process(self.process.pid)
                    children = parent.children(recursive=True)
                    
                    # 先终止子进程
                    for child in children:
                        try:
                            child.terminate()
                        except psutil.NoSuchProcess:
                            pass
                    
                    # 等待子进程结束
                    gone, still_alive = psutil.wait_procs(children, timeout=5)
                    
                    # 强制终止还在运行的子进程
                    for child in still_alive:
                        try:
                            child.kill()
                        except psutil.NoSuchProcess:
                            pass
                    
                    # 终止主进程
                    parent.terminate()
                    parent.wait(5)
                    
                    self.status = TrainStatus.STOPPED
                    log.info("训练进程已停止")
                    
                except Exception as e:
                    log.error(f"停止训练进程时出错: {e}")
                    # 强制 kill
                    try:
                        self.process.kill()
                    except:
                        pass
    
    def _read_logs(self):
        """读取训练日志"""
        if not self.process or not self.process.stdout:
            return
        
        try:
            for line in self.process.stdout:
                line = line.rstrip()
                if line:
                    # 存入队列和列表
                    try:
                        self.log_queue.put_nowait(line)
                    except queue.Full:
                        try:
                            self.log_queue.get_nowait()
                            self.log_queue.put_nowait(line)
                        except Exception:
                            pass
                    self.logs.append(line)
                    
                    # 限制日志数量
                    if len(self.logs) > self.max_logs:
                        self.logs.pop(0)
                    
                    # 尝试解析关键信息
                    self._parse_log_line(line)
                    
        except Exception as e:
            log.error(f"读取日志时出错: {e}")
    
    def _parse_log_line(self, line: str):
        """解析日志行提取关键信息"""
        import re
        
        # 解析 epoch
        match = re.search(r'epoch[:\s]+(\d+)', line, re.IGNORECASE)
        if match:
            self.current_epoch = int(match.group(1))
        
        # 解析 step
        match = re.search(r'step[:\s]+(\d+)', line, re.IGNORECASE)
        if match:
            self.current_step = int(match.group(1))
        
        # 解析 loss
        match = re.search(r'loss[=:\s]+([\d.]+)', line, re.IGNORECASE)
        if match:
            try:
                self.current_loss = float(match.group(1))
            except:
                pass
    
    def _monitor_process(self):
        """监控进程状态"""
        if not self.process:
            return
        
        try:
            return_code = self.process.wait()
            
            with self.lock:
                if return_code == 0:
                    self.status = TrainStatus.COMPLETED
                    log.info("训练任务已完成")
                elif self.status != TrainStatus.STOPPED:
                    self.status = TrainStatus.ERROR
                    log.error(f"训练进程异常退出，返回码: {return_code}")
                    
        except Exception as e:
            log.error(f"监控进程时出错: {e}")
    
    def get_logs(self, lines: int = 100) -> List[str]:
        """获取最近的日志行"""
        return self.logs[-lines:] if self.logs else []
    
    def get_status(self) -> Dict[str, Any]:
        """获取当前状态"""
        with self.lock:
            elapsed = None
            if self.start_time:
                elapsed = (datetime.now() - self.start_time).total_seconds()
            
            return {
                "status": self.status.value,
                "pid": self.process.pid if self.process else None,
                "running": self.status == TrainStatus.RUNNING,
                "elapsed_seconds": elapsed,
                "current_epoch": self.current_epoch,
                "current_step": self.current_step,
                "current_loss": self.current_loss,
                "log_count": len(self.logs),
            }
    
    def stop_all(self):
        """停止所有训练 (应用关闭时调用)"""
        self.stop_training()


# 全局训练服务实例
trainer_service = TrainerService()
