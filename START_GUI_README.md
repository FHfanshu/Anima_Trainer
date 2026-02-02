# 🚀 一键启动 Anima LoRA Trainer GUI

## 最简单的方法

### 方法一：双击运行 (Windows 推荐)
直接双击 **`start_gui.bat`** 文件即可！

### 方法二：命令行运行
```bash
cd Anima_Trainer
python start_gui.py
```

## 这个脚本会做什么？

脚本会自动完成以下6个步骤：

1. ✅ **检查 Python 环境** - 确保 Python 已安装
2. ✅ **检查 Python 依赖** - 自动安装 fastapi、uvicorn、pyyaml、psutil、rich
3. ✅ **检查 Node.js** - 确保 Node.js 和 npm 已安装
4. ✅ **安装前端依赖** - 运行 `npm install`
5. ✅ **构建前端** - 运行 `npm run build`
6. ✅ **启动服务** - 启动后端并自动打开浏览器

## 使用要求

### 必需软件
- **Python 3.8+** (必须)
- **Node.js 18+** (必须，用于构建前端)

### 可选软件
- **CUDA** (如果需要 GPU 训练)

## 启动方式对比

| 脚本 | 适用场景 | 特点 |
|------|---------|------|
| `start_gui.bat` / `start_gui.py` | **日常使用** | 一键完成所有检查和安装 |
| `gui/run_gui.py` | 开发调试 | 更多命令行参数选项 |
| `gui/launch.py` | 简化启动 | 仅启动，不做依赖检查 |

## 常见问题

### Q: 提示缺少 Node.js？
**A:** 请安装 Node.js: https://nodejs.org/
建议安装 LTS 版本 (18.x 或 20.x)

### Q: 安装前端依赖很慢？
**A:** 可以设置 npm 国内镜像加速：
```bash
npm config set registry https://registry.npmmirror.com
```

### Q: 端口被占用？
**A:** 脚本会自动尝试 28001-28020 范围内的端口，无需手动处理

### Q: 如何停止服务？
**A:** 在窗口中按 `Ctrl+C` 即可停止

### Q: 想重新构建前端？
**A:** 删除 `gui/frontend/dist` 文件夹，然后重新运行启动脚本

## 启动后

脚本会自动打开浏览器访问 `http://localhost:28000`

界面功能：
- 🏠 **首页** - 查看系统信息
- ⚙️ **训练配置** - 配置所有训练参数
- 📊 **训练监控** - 实时查看 Loss 曲线和日志
- 💾 **模型管理** - 管理训练好的模型

## 故障排除

如果启动失败，请检查：

1. **Python 版本** - 必须 3.8+
   ```bash
   python --version
   ```

2. **pip 是否可用**
   ```bash
   python -m pip --version
   ```

3. **Node.js 版本** - 必须 18+
   ```bash
   node --version
   npm --version
   ```

4. **端口占用**
   ```bash
   # Windows
   netstat -ano | findstr 28000
   ```

如果仍有问题，请查看终端输出的错误信息。
