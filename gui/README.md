# Anima LoRA Trainer GUI

🎨 基于 Vue3 + Vite + FastAPI 的 Anima 模型 LoRA 训练图形界面

## 特性

- ✨ **中文界面** - 完整的中文 UI，操作简单直观
- 🌙 **深色模式** - 支持亮色/深色主题切换
- 📊 **实时图表** - 使用 ECharts 展示 Loss 曲线和学习率变化
- 💻 **实时监控** - 实时训练日志和 GPU 显存监控
- ⚙️ **完整配置** - 支持所有 train.py 参数的可视化配置
- 💾 **预设管理** - 内置多种训练预设（角色/风格/概念/快速测试）
- 🗂️ **Checkpoint 管理** - 可视化的模型管理和导出

## 快速开始

### 1. 安装依赖

确保已安装 Python 依赖：

```bash
pip install fastapi uvicorn pyyaml psutil
```

### 2. 启动 GUI

```bash
cd Anima_Trainer
gui\launch.py
```

或者使用命令行参数：

```bash
python gui/run_gui.py --port 28000
```

启动后会自动打开浏览器访问 `http://localhost:28000`

## 开发模式

### 前端开发

```bash
cd gui/frontend
npm install
npm run dev
```

前端开发服务器运行在 `http://localhost:3000`，API 请求会自动代理到 `http://localhost:28000`

### 后端开发

```bash
python gui/run_gui.py --dev
```

启用热重载模式，方便调试

## 使用说明

### 配置训练

1. 进入"配置训练"页面
2. 选择或加载预设（角色/风格/概念/快速测试）
3. 修改参数：
   - **模型设置**: 模型路径、LoRA 类型、Rank/Alpha
   - **数据集**: 数据目录、分辨率、数据增强
   - **训练参数**: Epoch/Steps、Batch Size、学习率
   - **优化器**: 8-bit AdamW / muon
4. 点击"保存配置"或直接"开始训练"

### 训练监控

在"训练控制台"页面可以：
- 查看实时 Loss 曲线和学习率曲线
- 监控 GPU 显存使用
- 查看实时训练日志
- 控制训练（暂停/停止）
- 查看验证图像

### 模型管理

在"模型管理"页面可以：
- 查看所有训练的 Checkpoint
- 导出 LoRA 模型
- 删除不需要的模型

## 端口说明

- **28000**: 主服务端口（HTTP + API）
- **3000**: 前端开发服务器（仅开发模式）

如果 28000 被占用，会自动尝试 28001-28020 范围内的端口。

## 技术栈

### 后端
- **FastAPI** - 高性能异步 Web 框架
- **Uvicorn** - ASGI 服务器
- **PYYAML** - YAML 配置处理
- **psutil** - 系统进程管理

### 前端
- **Vue 3** - 渐进式 JavaScript 框架
- **Vite** - 极速构建工具
- **TypeScript** - 类型安全的 JavaScript
- **Element Plus** - 企业级 UI 组件库
- **ECharts** - 数据可视化图表
- **Pinia** - 状态管理
- **Axios** - HTTP 客户端

## 项目结构

```
gui/
├── backend/              # FastAPI 后端
│   ├── app.py           # FastAPI 应用主入口
│   ├── log.py           # 日志模块
│   ├── api/             # API 路由
│   │   ├── config.py    # 配置管理 API
│   │   ├── train.py     # 训练控制 API
│   │   └── system.py    # 系统信息 API
│   └── services/        # 业务服务
│       └── trainer.py   # 训练进程管理
├── frontend/            # Vue3 前端
│   ├── src/
│   │   ├── api/         # API 客户端
│   │   ├── components/  # 可复用组件
│   │   ├── stores/      # Pinia 状态管理
│   │   ├── views/       # 页面视图
│   │   └── types/       # TypeScript 类型
│   ├── package.json
│   └── vite.config.ts
├── launch.py            # 一键启动脚本
└── run_gui.py           # 命令行启动脚本
```

## API 文档

启动服务后访问：`http://localhost:28000/docs`

主要 API：
- `GET /api/config/default` - 获取默认配置
- `POST /api/train/start` - 开始训练
- `POST /api/train/stop` - 停止训练
- `GET /api/train/status` - 获取训练状态
- `GET /api/train/logs` - 获取训练日志
- `GET /api/train/metrics` - 获取训练指标
- `GET /api/system/info` - 获取系统信息
- `GET /api/system/gpu_status` - 获取 GPU 状态

## 注意事项

1. **依赖安装**: 首次使用前需要安装前端依赖（`npm install`）
2. **构建**: 生产环境需要构建前端（`npm run build`）
3. **端口**: 默认使用 28000 端口，如果被占用会自动尝试其他端口
4. **GPU**: 需要 NVIDIA GPU 和 CUDA 环境才能训练
5. **内存**: 建议至少 16GB 显存（RTX 3090/4090 最佳）

## 故障排除

### 前端构建失败
```bash
cd gui/frontend
npm install
npm run build
```

### 端口被占用
脚本会自动尝试其他端口，或手动指定：
```bash
python gui/run_gui.py --port 28001
```

### 无法启动训练
检查：
1. 数据集路径是否正确
2. 是否有图片文件
3. Python 环境是否正确

## 许可证

MIT License
