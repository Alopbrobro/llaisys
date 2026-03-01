# llaisys Docker 部署指南

本指南帮助你在算力平台上使用 Docker 部署和运行 llaisys。

## 前置要求

- Docker (>= 20.10)
- NVIDIA Docker Runtime（用于 GPU 支持）
- NVIDIA GPU 和驱动（可选，用于 GPU 加速）

## 快速开始

### 1. 构建 Docker 镜像

```bash
# 方式 1: 使用便捷脚本
./docker.sh build

# 方式 2: 直接使用 docker 命令
docker build -t llaisys:latest .
```

构建过程大约需要 10-20 分钟，包括：
- 安装系统依赖
- 安装 xmake 构建工具
- 编译 C++ 库（支持 NVIDIA GPU）
- 安装 Python 依赖

### 2. 运行容器

```bash
# 方式 1: 使用便捷脚本（推荐）
./docker.sh run

# 方式 2: 直接使用 docker 命令
docker run -it --rm --gpus all -v $(pwd):/workspace/llaisys llaisys:latest
```

### 3. 在容器中测试

进入容器后：

```bash
# 测试 Python 导入
python3 -c "import llaisys; print(llaisys.__version__)"

# 运行测试套件
cd /workspace/llaisys
python3 -m pytest test/

# 运行推理测试
python3 test/test_infer.py
```

## 常用命令

### 构建镜像
```bash
./docker.sh build
```

### 运行交互式容器
```bash
./docker.sh run
```

### 运行测试
```bash
./docker.sh test
```

### 清理 Docker 资源
```bash
./docker.sh clean
```

### 推送镜像到仓库
```bash
# 先登录到你的镜像仓库
docker login registry.example.com

# 推送镜像
./docker.sh push registry.example.com/your-username/llaisys
```

## 在算力平台上使用

### AutoDL / 智星云 / 恒源云等平台

1. **上传代码**
   ```bash
   # 在本地构建镜像并推送到 Docker Hub
   docker build -t yourusername/llaisys:latest .
   docker push yourusername/llaisys:latest
   ```

2. **在平台上拉取镜像**
   ```bash
   docker pull yourusername/llaisys:latest
   ```

3. **启动容器**
   ```bash
   docker run -it --gpus all \
       -v /your/data:/workspace/data \
       yourusername/llaisys:latest
   ```

### Kubernetes 部署

创建 deployment.yaml：

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: llaisys-inference
spec:
  containers:
  - name: llaisys
    image: yourusername/llaisys:latest
    resources:
      limits:
        nvidia.com/gpu: 1
    volumeMounts:
    - name: workspace
      mountPath: /workspace/data
  volumes:
  - name: workspace
    hostPath:
      path: /path/to/your/data
```

应用配置：
```bash
kubectl apply -f deployment.yaml
kubectl exec -it llaisys-inference -- /bin/bash
```

## 镜像定制

### 使用不同的 CUDA 版本

修改 [Dockerfile](Dockerfile) 第一行：

```dockerfile
# CUDA 12.1
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# CUDA 11.7
FROM nvidia/cuda:11.7.0-cudnn8-devel-ubuntu22.04
```

### 添加额外的 Python 依赖

修改 [Dockerfile](Dockerfile)：

```dockerfile
RUN pip install torch>=2.4.0 transformers accelerate \
    numpy pandas scikit-learn jupyter
```

### 仅使用 CPU

构建时禁用 NVIDIA GPU 支持：

```dockerfile
# 修改 Dockerfile 中的编译行
RUN /root/.local/bin/xmake f -m release && \
    /root/.local/bin/xmake -j$(nproc) && \
    /root/.local/bin/xmake install
```

## 故障排查

### GPU 不可用

检查 NVIDIA Docker 是否正确安装：
```bash
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### 编译失败

查看 xmake 日志：
```bash
docker build --no-cache -t llaisys:latest .
```

### 内存不足

增加 Docker 内存限制：
```bash
docker run -it --gpus all --memory=16g llaisys:latest
```

## 性能优化建议

1. **多阶段构建**：减小最终镜像大小
2. **缓存层**：合理安排 Dockerfile 指令顺序
3. **并行编译**：xmake 已启用 `-j$(nproc)`
4. **挂载数据**：使用 volume 而非 COPY 大文件

## 支持

如有问题，请查看：
- 项目主 README: [README.md](README.md)
- 提交 Issue: GitHub Issues
