"""
一键启动脚本
整合后端启动和前端构建
"""

import os
import sys
import subprocess
import webbrowser
import time
from pathlib import Path

def run_command(cmd, cwd=None, shell=False):
    """运行命令并返回进程"""
    return subprocess.Popen(
        cmd,
        cwd=cwd,
        shell=shell,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

def main():
    print("🎨 Anima LoRA Trainer GUI 启动脚本")
    print("=" * 50)
    
    # 项目路径
    project_root = Path(__file__).parent.parent
    gui_dir = project_root / "gui"
    frontend_dir = gui_dir / "frontend"
    backend_dir = gui_dir / "backend"
    
    # 检查是否需要构建前端
    dist_dir = frontend_dir / "dist"
    if not dist_dir.exists() or not (dist_dir / "index.html").exists():
        print("📦 前端未构建，正在构建...")
        print("   安装依赖...")
        
        # 安装依赖
        result = subprocess.run(
            ["npm", "install"],
            cwd=str(frontend_dir),
            shell=True,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"❌ 安装依赖失败: {result.stderr}")
            return
        
        print("   构建前端...")
        result = subprocess.run(
            ["npm", "run", "build"],
            cwd=str(frontend_dir),
            shell=True,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"❌ 构建失败: {result.stderr}")
            return
        
        print("✅ 前端构建完成")
    
    # 检查端口
    import socket
    port = 28000
    
    def check_port_available(p, host="127.0.0.1"):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind((host, p))
                return True
        except OSError:
            return False
    
    if not check_port_available(port):
        print(f"⚠️  端口 {port} 被占用，尝试其他端口...")
        for p in range(28001, 28021):
            if check_port_available(p):
                port = p
                break
        else:
            print("❌ 无法找到可用端口")
            return
    
    print(f"📡 将使用端口: {port}")
    
    # 设置环境变量
    os.environ["ANIMA_GUI_HOST"] = "127.0.0.1"
    os.environ["ANIMA_GUI_PORT"] = str(port)
    os.environ["ANIMA_GUI_DEV"] = "0"
    
    # 启动后端
    print("🚀 启动后端服务...")
    try:
        import uvicorn
        print(f"   服务地址: http://127.0.0.1:{port}")
        print("   按 Ctrl+C 停止服务")
        print("-" * 50)
        
        # 自动打开浏览器
        time.sleep(1)
        webbrowser.open(f"http://127.0.0.1:{port}")
        
        # 启动服务
        uvicorn.run(
            "gui.backend.app:app",
            host="127.0.0.1",
            port=port,
            reload=False,
            log_level="info"
        )
        
    except ImportError:
        print("❌ 请先安装依赖: pip install fastapi uvicorn")
        return
    except KeyboardInterrupt:
        print("\n👋 服务已停止")

if __name__ == "__main__":
    main()
