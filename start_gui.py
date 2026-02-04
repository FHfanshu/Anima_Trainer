"""
Anima LoRA Trainer - 真正的一键启动脚本
自动检查并安装所有依赖，然后启动 GUI
"""

import os
import sys
import subprocess
import webbrowser
import time
import socket
from pathlib import Path
import importlib
import venv

# 颜色代码
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def print_header(text):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(60)}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.END}\n")

def print_success(text):
    print(f"{Colors.GREEN}[OK] {text}{Colors.END}")

def print_error(text):
    print(f"{Colors.RED}[ERROR] {text}{Colors.END}")

def print_warning(text):
    print(f"{Colors.YELLOW}[WARN] {text}{Colors.END}")

def print_info(text):
    print(f"{Colors.BLUE}[INFO] {text}{Colors.END}")

def print_step(step_num, total_steps, text):
    print(f"{Colors.CYAN}[Step {step_num}/{total_steps}] {text}...{Colors.END}")

def check_python_package(package_name, import_name=None):
    """检查 Python 包是否安装"""
    if import_name is None:
        import_name = package_name
    try:
        importlib.import_module(import_name)
        return True
    except ImportError:
        return False

def check_torch_version(expected_version: str, expected_cuda: str) -> bool:
    try:
        import torch  # type: ignore
    except ImportError:
        return False
    version = getattr(torch, "__version__", "")
    cuda_version = getattr(torch.version, "cuda", None)
    if not version.startswith(expected_version):
        return False
    if not cuda_version:
        return False
    return str(cuda_version).startswith(expected_cuda)

def in_venv() -> bool:
    return getattr(sys, "base_prefix", sys.prefix) != sys.prefix

def get_venv_python(venv_dir: Path) -> Path:
    if sys.platform == "win32":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"

def ensure_venv(project_root: Path) -> None:
    """Auto-create and re-exec into repo venv.

    - Creates `.venv/` in repo root if missing
    - Re-runs this script using the venv Python
    - Set `ANIMA_USE_SYSTEM_PYTHON=1` to disable
    """
    if os.environ.get("ANIMA_USE_SYSTEM_PYTHON", "0") == "1":
        return
    if in_venv():
        return
    if os.environ.get("ANIMA_VENV_BOOTSTRAPPED", "0") == "1":
        return

    venv_dir = project_root / ".venv"
    venv_python = get_venv_python(venv_dir)

    if not venv_python.exists():
        print("[INFO] Creating virtual environment: .venv")
        venv.EnvBuilder(with_pip=True).create(str(venv_dir))

    if not venv_python.exists():
        raise RuntimeError("Failed to create venv (.venv)")

    os.environ["ANIMA_VENV_BOOTSTRAPPED"] = "1"
    os.execv(
        str(venv_python),
        [str(venv_python), str(Path(__file__).resolve()), *sys.argv[1:]],
    )

def install_python_package(package):
    """安装 Python 包"""
    print_info(f"正在安装 {package}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package], 
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except:
        return False

def check_command(command):
    """检查命令是否可用"""
    try:
        subprocess.run([command, "--version"], capture_output=True, check=True, shell=True, encoding="utf-8", errors="replace")
        return True
    except:
        return False

def ensure_uv_available() -> bool:
    """Ensure uv is available in this environment.

    We install it into the active venv (recommended) when missing.
    """
    if check_python_package("uv"):
        return True
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-U", "uv"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return False
    return check_python_package("uv")

def uv_pip_install(packages, extra_index_url: str | None = None) -> bool:
    if not ensure_uv_available():
        return False
    cmd = [sys.executable, "-m", "uv", "pip", "install"]
    if extra_index_url:
        cmd += ["--extra-index-url", extra_index_url]
    cmd += list(packages)
    try:
        subprocess.check_call(cmd)
        return True
    except Exception:
        return False

def check_port_available(port, host="127.0.0.1"):
    """检查端口是否可用"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((host, port))
            return True
    except OSError:
        return False

def run_npm(cmd, cwd, description):
    """运行 npm 命令并处理错误"""
    print_info(f"执行: npm {' '.join(cmd[1:])}")
    try:
        result = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            shell=True,
            encoding="utf-8",
            errors="replace"
        )
        
        if result.returncode != 0:
            print_error(f"{description}失败")
            if result.stderr:
                print(f"错误信息: {result.stderr}")
            return False
        
        print_success(f"{description}完成")
        return True
    except Exception as e:
        print_error(f"{description}时出错: {e}")
        return False

def main():
    project_root = Path(__file__).parent
    ensure_venv(project_root)

    print_header("Anima LoRA Trainer Launcher")
    
    # 项目路径
    project_root = Path(__file__).parent
    gui_dir = project_root / "gui"
    frontend_dir = gui_dir / "frontend"
    
    # 总共6个步骤
    total_steps = 6
    current_step = 0
    
    # ========== 步骤 1: 检查 Python 环境 ==========
    current_step += 1
    print_step(current_step, total_steps, "检查 Python 环境")
    
    print_info(f"Python 版本: {sys.version.split()[0]}")
    print_info(f"Python 路径: {sys.executable}")
    print_success("Python 环境检查通过")
    
    # ========== 步骤 2: 检查并安装 Python 依赖 ==========
    current_step += 1
    print_step(current_step, total_steps, "检查 Python 依赖")
    
    # 必需依赖 (GUI + system info)
    expected_torch_version = "2.10.0"
    expected_cuda_version = "13.0"  # cu130
    torch_extra_index = "https://download.pytorch.org/whl/cu130"

    required_packages = [
        {"spec": "fastapi", "import": "fastapi"},
        {"spec": "uvicorn", "import": "uvicorn"},
        {"spec": "pyyaml", "import": "yaml"},
        {"spec": "psutil", "import": "psutil"},
        {"spec": "rich", "import": "rich"},
        {"spec": "python-multipart", "import": "multipart"},
        {"spec": "diffusers", "import": "diffusers"},
        {"spec": "torch==2.10.0+cu130", "import": "torch", "torch": True},
    ]

    missing_general = []
    need_torch = False

    for item in required_packages:
        spec = item["spec"]
        import_name = item["import"]
        if item.get("torch"):
            if check_python_package(import_name) and check_torch_version(expected_torch_version, expected_cuda_version):
                continue
            need_torch = True
            print_warning(f"torch 版本不匹配或未安装，需 {expected_torch_version}+cu130")
            continue

        if not check_python_package(import_name):
            missing_general.append(spec)
            print_warning(f"缺少依赖: {spec}")

    if missing_general or need_torch:
        total_missing = len(missing_general) + (1 if need_torch else 0)
        print_info(f"正在安装 {total_missing} 个缺失的依赖 (使用 uv)...")

        if missing_general:
            if not uv_pip_install(missing_general):
                print_error("安装依赖失败")
                print(f"\n{Colors.RED}请手动运行: {sys.executable} -m pip install -U uv && {sys.executable} -m uv pip install {' '.join(missing_general)}{Colors.END}")
                input("\n按 Enter 键退出...")
                return
            print_success("基础依赖安装完成")

        if need_torch:
            if not uv_pip_install(["torch==2.10.0+cu130"], extra_index_url=torch_extra_index):
                print_error("安装 torch 失败")
                print(f"\n{Colors.RED}请手动运行: {sys.executable} -m uv pip install --extra-index-url {torch_extra_index} torch==2.10.0+cu130{Colors.END}")
                input("\n按 Enter 键退出...")
                return
            print_success("torch 安装完成")
    else:
        print_success("所有 Python 依赖已安装")
    
    # ========== 步骤 3: 检查 Node.js ==========
    current_step += 1
    print_step(current_step, total_steps, "检查 Node.js 环境")
    
    if not check_command("node"):
        print_error("未检测到 Node.js")
        print_warning("请先安装 Node.js: https://nodejs.org/")
        print_info("建议安装 LTS 版本 (18.x 或 20.x)")
        webbrowser.open("https://nodejs.org/")
        input("\n安装完成后按 Enter 键继续...")
        
        # 再次检查
        if not check_command("node"):
            print_error("仍然未检测到 Node.js，退出")
            return
    
    # 检查 Node.js 版本
    try:
        result = subprocess.run(["node", "--version"], capture_output=True, text=True, encoding="utf-8", errors="replace")
        node_version = result.stdout.strip()
        print_success(f"Node.js 版本: {node_version}")
    except:
        print_warning("无法获取 Node.js 版本")
    
    # 检查 npm - 尝试多种方式
    npm_available = False
    npm_cmd = None
    
    # 尝试检测 npm
    for cmd in ["npm", "npm.cmd", "npm.exe"]:
        if check_command(cmd):
            npm_available = True
            npm_cmd = cmd
            break
    
    # 如果还是找不到，尝试从 node 路径推断
    if not npm_available:
        try:
            result = subprocess.run(["where", "node"], capture_output=True, text=True, shell=True, encoding="utf-8", errors="replace")
            if result.returncode == 0:
                node_path = result.stdout.strip().split('\n')[0].strip()
                node_dir = Path(node_path).parent
                npm_path = node_dir / "npm.cmd"
                if npm_path.exists():
                    npm_available = True
                    npm_cmd = str(npm_path)
                    print_success(f"找到 npm: {npm_cmd}")
        except:
            pass
    
    # 如果找不到 npm，检查是否已经有构建好的前端
    dist_dir = frontend_dir / "dist"
    if not npm_available and dist_dir.exists() and (dist_dir / "index.html").exists():
        print_warning("未检测到 npm，但发现已构建的前端")
        print_info("将跳过前端构建步骤，直接启动服务")
        npm_cmd = None
    elif not npm_available or npm_cmd is None:
        print_error("未检测到 npm")
        print_warning("npm 应该随 Node.js 一起安装")
        print_info("尝试解决方法:")
        print_info("  1. 重新安装 Node.js (推荐): https://nodejs.org/")
        print_info("  2. 确保安装时勾选 'Add to PATH'")
        print_info("  3. 重启电脑后再次尝试")
        webbrowser.open("https://nodejs.org/")
        input("\n按 Enter 键退出...")
        return
    else:
        print_success(f"npm 可用: {npm_cmd}")
    
    # ========== 步骤 4 & 5: 安装前端依赖 & 构建前端 ==========
    dist_dir = frontend_dir / "dist"
    
    if npm_cmd is None:
        # 跳过前端依赖安装（已有构建好的前端）
        current_step += 1
        print_step(current_step, total_steps, "检查前端依赖 (跳过)")
        print_info("使用已构建的前端，跳过 npm 步骤")
        current_step += 1
        print_step(current_step, total_steps, "构建前端 (跳过)")
        print_info("使用已构建的前端")
    else:
        current_step += 1
        print_step(current_step, total_steps, "安装前端依赖")
        
        node_modules = frontend_dir / "node_modules"
        if not node_modules.exists():
            print_info("首次运行，需要安装前端依赖 (可能需要几分钟)...")
            print_info(f"工作目录: {frontend_dir}")
            
            # 使用 npm ci 如果有 package-lock.json，否则用 npm install
            if (frontend_dir / "package-lock.json").exists():
                cmd = [npm_cmd, "ci"]
            else:
                cmd = [npm_cmd, "install"]
            
            if not run_npm(cmd, frontend_dir, "安装前端依赖"):
                input("\n按 Enter 键退出...")
                return
        else:
            print_success("前端依赖已安装")
        
        # ========== 步骤 5: 构建前端 ==========
        current_step += 1
        print_step(current_step, total_steps, "构建前端")
        
        package_json = frontend_dir / "package.json"
        
        # 检查是否需要重新构建
        need_build = False
        if not dist_dir.exists():
            need_build = True
            print_info("前端构建目录不存在")
        elif not (dist_dir / "index.html").exists():
            need_build = True
            print_info("前端入口文件不存在")
        elif package_json.exists():
            # 检查 package.json 是否比 dist 新
            dist_mtime = dist_dir.stat().st_mtime
            package_mtime = package_json.stat().st_mtime
            if package_mtime > dist_mtime:
                need_build = True
                print_info("检测到前端代码有更新")
        
        if need_build:
            print_info("正在构建前端 (生产模式)...")
            if not run_npm([npm_cmd, "run", "build"], frontend_dir, "构建前端"):
                input("\n按 Enter 键退出...")
                return
        else:
            print_success("前端已构建 (无需更新)")
    
    # ========== 步骤 6: 启动服务 ==========
    current_step += 1
    print_step(current_step, total_steps, "启动服务")
    
    # 检查端口
    port = 28000
    if not check_port_available(port):
        print_warning(f"端口 {port} 被占用")
        for p in range(28001, 28021):
            if check_port_available(p):
                port = p
                print_success(f"找到可用端口: {p}")
                break
        else:
            print_error("无法找到可用端口 (28000-28020)")
            print_warning("请关闭占用这些端口的程序")
            input("\n按 Enter 键退出...")
            return
    
    print_success(f"使用端口: {port}")
    
    # 设置环境变量
    os.environ["ANIMA_GUI_HOST"] = "127.0.0.1"
    os.environ["ANIMA_GUI_PORT"] = str(port)
    os.environ["ANIMA_GUI_DEV"] = "0"
    
    # 打印启动信息
    print_header("Starting Anima LoRA Trainer GUI")
    print(f"{Colors.GREEN}Server URL: http://127.0.0.1:{port}{Colors.END}")
    print(f"{Colors.CYAN}Press Ctrl+C to stop{Colors.END}")
    print(f"{Colors.YELLOW}Opening browser in 2 seconds...{Colors.END}\n")
    
    # 延迟后打开浏览器
    time.sleep(2)
    webbrowser.open(f"http://127.0.0.1:{port}")
    
    # 启动服务
    try:
        import uvicorn
        uvicorn.run(
            "gui.backend.app:app",
            host="127.0.0.1",
            port=port,
            reload=False,
            log_level="info"
        )
    except ImportError:
        print_error("Missing uvicorn, please run: pip install uvicorn")
        input("\nPress Enter to exit...")
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Server stopped, goodbye!{Colors.END}")
        time.sleep(1)

if __name__ == "__main__":
    # Windows 下设置控制台为 UTF-8
    if sys.platform == "win32":
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
    
    try:
        main()
    except Exception as e:
        print(f"\n{Colors.RED}发生错误: {e}{Colors.END}")
        import traceback
        traceback.print_exc()
        input("\n按 Enter 键退出...")
