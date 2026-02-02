import os
import sys

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import uvicorn
import webbrowser
from pathlib import Path


def check_port_available(port: int, host: str = "127.0.0.1") -> bool:
    """检查端口是否可用"""
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((host, port))
            return True
    except OSError:
        return False


def find_available_port(start: int, end: int, host: str = "127.0.0.1") -> int:
    """查找可用端口"""
    for port in range(start, end + 1):
        if check_port_available(port, host):
            return port
    return None


def main():
    """启动 Anima LoRA Trainer GUI"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Anima LoRA Trainer GUI")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="监听地址")
    parser.add_argument("--port", type=int, default=28000, help="监听端口")
    parser.add_argument("--listen", action="store_true", help="监听所有地址 (0.0.0.0)")
    parser.add_argument("--dev", action="store_true", help="开发模式 (热重载)")
    parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    
    args = parser.parse_args()
    
    if args.listen:
        args.host = "0.0.0.0"
    
    # 检查端口
    if not check_port_available(args.port, args.host):
        print(f"端口 {args.port} 被占用，正在查找可用端口...")
        available_port = find_available_port(28001, 28020, args.host)
        if available_port:
            args.port = available_port
            print(f"切换到端口 {args.port}")
        else:
            print("错误: 无法找到可用端口 (28000-28020)")
            sys.exit(1)
    
    # 设置环境变量
    os.environ["ANIMA_GUI_HOST"] = args.host
    os.environ["ANIMA_GUI_PORT"] = str(args.port)
    os.environ["ANIMA_GUI_DEV"] = "1" if args.dev else "0"
    
    url = f"http://{args.host}:{args.port}"
    print(f"🎨 Anima LoRA Trainer GUI 启动中...")
    print(f"📡 地址: {url}")
    print(f"🔧 模式: {'开发' if args.dev else '生产'}")
    print(f"📁 工作目录: {project_root}")
    
    # 自动打开浏览器
    if not args.no_browser and not args.dev and sys.platform == "win32":
        webbrowser.open(url)
    
    # 启动 FastAPI
    uvicorn.run(
        "gui.backend.app:app",
        host=args.host,
        port=args.port,
        reload=args.dev,
        log_level="info" if args.dev else "error"
    )


if __name__ == "__main__":
    main()
