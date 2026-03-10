#!/bin/bash
# =============================================================
# 沐曦 (MetaX) C500 平台一键部署脚本
# =============================================================
# 适用场景：租用沐曦算力后，在 PyTorch 预载镜像中执行
#
# 前提条件：
#   - 预载镜像选择：PyTorch（含 MXMACA SDK）
#   - MXMACA SDK 已预装在 /opt/mxmaca/
#   - Python 3.8+ 已可用
#
# 用法：
#   chmod +x scripts/deploy_metax.sh
#   ./scripts/deploy_metax.sh [模型路径]
#
# 示例：
#   ./scripts/deploy_metax.sh ./quantized_model
#   ./scripts/deploy_metax.sh ./quantized_model_int4
#   ./scripts/deploy_metax.sh Qwen/Qwen2-1.5B
# =============================================================

set -e

# ── 颜色输出 ──────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC}  $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }
step()  { echo -e "\n${CYAN}═══════════════════════════════════════${NC}"; echo -e "${CYAN}  $1${NC}"; echo -e "${CYAN}═══════════════════════════════════════${NC}"; }

MODEL_PATH="${1:-./quantized_model}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# ── Step 1: 环境检测 ─────────────────────────────────────────
step "Step 1/6: 环境检测"

# 检查 MXMACA SDK
if [ -d "/opt/mxmaca/include" ]; then
    info "MXMACA SDK 已就绪: /opt/mxmaca/"
    if [ -f "/opt/mxmaca/bin/mxcc" ]; then
        info "mxcc 编译器: $(/opt/mxmaca/bin/mxcc --version 2>&1 | head -1 || echo '已安装')"
    fi
else
    error "未检测到 MXMACA SDK (/opt/mxmaca/ 不存在)"
    error "请确认选择了 PyTorch 预载镜像（含 MXMACA SDK）"
    exit 1
fi

# 检查 maca_runtime
if [ -f "/opt/mxmaca/lib/libmaca_runtime.so" ]; then
    info "maca_runtime: OK"
else
    warn "未找到 libmaca_runtime.so，将尝试查找..."
    LIB_MACA=$(find /opt/mxmaca -name "libmaca_runtime*" 2>/dev/null | head -1)
    if [ -n "$LIB_MACA" ]; then
        info "找到 maca_runtime: $LIB_MACA"
    else
        error "未找到 maca_runtime，请检查 MXMACA SDK 安装"
        exit 1
    fi
fi

# 检查 mxBLAS
if [ -f "/opt/mxmaca/lib/libmxblas.so" ]; then
    info "mxBLAS: OK"
else
    warn "未找到 libmxblas.so，尝试查找..."
    LIB_BLAS=$(find /opt/mxmaca -name "libmxblas*" 2>/dev/null | head -1)
    if [ -n "$LIB_BLAS" ]; then
        info "找到 mxBLAS: $LIB_BLAS"
    else
        error "未找到 mxBLAS，linear/self_attention 算子将无法工作"
        exit 1
    fi
fi

# 检查 GPU
info "检测沐曦 GPU..."
if command -v mx-smi &>/dev/null; then
    mx-smi 2>&1 | head -20
elif command -v maca-smi &>/dev/null; then
    maca-smi 2>&1 | head -20
else
    warn "未找到 mx-smi/maca-smi 工具，跳过 GPU 检测"
    warn "（不影响编译，可稍后手动验证）"
fi

# 检查 Python
PYTHON_CMD=""
if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
elif command -v python &>/dev/null; then
    PYTHON_CMD="python"
else
    error "未找到 Python，请安装 Python 3.8+"
    exit 1
fi
info "Python: $($PYTHON_CMD --version)"

# ── Step 2: 安装 xmake ──────────────────────────────────────
step "Step 2/6: 安装 xmake 构建系统"

if command -v xmake &>/dev/null; then
    info "xmake 已安装: $(xmake --version | head -1)"
else
    info "正在安装 xmake..."
    # 尝试多种安装方式
    if curl -fsSL https://xmake.io/shget.text -o /tmp/xmake_install.sh 2>/dev/null; then
        bash /tmp/xmake_install.sh
        export PATH="$HOME/.local/bin:$PATH"
    elif command -v pip3 &>/dev/null; then
        pip3 install xmake
    else
        error "无法安装 xmake，请手动安装: https://xmake.io"
        exit 1
    fi

    if command -v xmake &>/dev/null; then
        info "xmake 安装成功: $(xmake --version | head -1)"
    else
        # 尝试加载环境
        source "$HOME/.xmake/profile" 2>/dev/null || true
        export PATH="$HOME/.local/bin:$PATH"
        if command -v xmake &>/dev/null; then
            info "xmake 安装成功: $(xmake --version | head -1)"
        else
            error "xmake 安装后无法找到，请手动将其加入 PATH"
            exit 1
        fi
    fi
fi

# ── Step 3: 编译 llaisys ────────────────────────────────────
step "Step 3/6: 编译 llaisys (MetaX GPU)"

cd "$PROJECT_DIR"

info "配置构建（启用沐曦 GPU）..."
xmake f --metax-gpu=true -c -y

info "开始编译..."
xmake build -j$(nproc)

info "编译成功！"

# 验证生成的共享库
if [ -f "lib/libllaisys.so" ]; then
    info "共享库: lib/libllaisys.so ($(du -h lib/libllaisys.so | cut -f1))"
else
    LIB_SO=$(find build -name "libllaisys.so" 2>/dev/null | head -1)
    if [ -n "$LIB_SO" ]; then
        info "共享库: $LIB_SO"
    else
        error "未找到 libllaisys.so，编译可能不完整"
        exit 1
    fi
fi

# ── Step 4: 安装 Python 依赖 ────────────────────────────────
step "Step 4/6: 安装 Python 依赖"

cd "$PROJECT_DIR/python"

# 安装 Python 包
info "安装 llaisys Python 包..."
$PYTHON_CMD -m pip install -e . --quiet 2>&1 | tail -5

# 验证关键依赖
info "验证依赖..."
$PYTHON_CMD -c "import transformers; print(f'  transformers: {transformers.__version__}')" 2>/dev/null || {
    warn "transformers 未安装，正在安装..."
    $PYTHON_CMD -m pip install transformers --quiet
}

$PYTHON_CMD -c "import torch; print(f'  torch: {torch.__version__}')" 2>/dev/null || {
    warn "PyTorch 未安装，请从沐曦镜像源安装适配版 PyTorch"
}

$PYTHON_CMD -c "import safetensors; print(f'  safetensors: {safetensors.__version__}')" 2>/dev/null || {
    warn "safetensors 未安装，正在安装..."
    $PYTHON_CMD -m pip install safetensors --quiet
}

# 安装 Web 服务器依赖
$PYTHON_CMD -m pip install fastapi uvicorn --quiet 2>&1 | tail -3

# ── Step 5: 部署共享库 ──────────────────────────────────────
step "Step 5/6: 部署共享库到 Python 包"

cd "$PROJECT_DIR"

# xmake install 会自动复制 .so 到 python/llaisys/libllaisys/
xmake install -o . 2>&1 | tail -5 || true

# 手动确认
if [ -f "python/llaisys/libllaisys/libllaisys.so" ]; then
    info "libllaisys.so 已部署到 Python 包"
else
    warn "尝试手动复制 .so 文件..."
    cp lib/libllaisys.so python/llaisys/libllaisys/ 2>/dev/null || \
    cp build/linux/x86_64/release/libllaisys.so python/llaisys/libllaisys/ 2>/dev/null || {
        error "无法复制 libllaisys.so 到 Python 包，请手动复制"
    }
fi

# ── Step 6: 验证 & 启动 ─────────────────────────────────────
step "Step 6/6: 快速验证"

cd "$PROJECT_DIR"

info "运行基础测试..."
$PYTHON_CMD -c "
import llaisys
print('  llaisys 模块加载: OK')
print(f'  DeviceType.METAX = {llaisys.DeviceType.METAX}')

# 测试 RuntimeAPI
rt = llaisys.RuntimeAPI(llaisys.DeviceType.METAX)
count = rt.getDeviceCount()
print(f'  MetaX GPU 数量: {count}')

if count > 0:
    rt.setDevice(0)
    print('  setDevice(0): OK')
    print('  沐曦 C500 GPU 就绪！')
else:
    print('  [WARN] 未检测到 MetaX GPU，请检查驱动')
" 2>&1 || {
    warn "基础测试未通过（可能需要检查 GPU 驱动）"
}

# ── 输出摘要 ─────────────────────────────────────────────────
echo ""
echo -e "${GREEN}══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  部署完成！${NC}"
echo -e "${GREEN}══════════════════════════════════════════════════════════${NC}"
echo ""
echo "  启动推理服务器："
echo -e "    ${CYAN}cd $PROJECT_DIR/python${NC}"
echo -e "    ${CYAN}$PYTHON_CMD -m server.app --model $MODEL_PATH --device metax${NC}"
echo ""
echo "  启动 CLI 聊天："
echo -e "    ${CYAN}$PYTHON_CMD -m server.chat_cli${NC}"
echo ""
echo "  运行测试："
echo -e "    ${CYAN}cd $PROJECT_DIR && $PYTHON_CMD -m pytest test/${NC}"
echo ""
echo -e "${YELLOW}  注意：首次启动需要加载模型权重，可能需要几分钟${NC}"
echo ""
