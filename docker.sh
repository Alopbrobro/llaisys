#!/bin/bash

# Docker 构建和运行脚本

set -e

# 配置变量
IMAGE_NAME="llaisys"
IMAGE_TAG="latest"
CONTAINER_NAME="llaisys-dev"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查 Docker 是否安装
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker 未安装，请先安装 Docker"
        exit 1
    fi
    print_info "Docker 版本: $(docker --version)"
}

# 检查 NVIDIA Docker 支持
check_nvidia_docker() {
    if ! docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
        print_warn "NVIDIA Docker 支持未启用或 GPU 不可用"
        print_warn "将以 CPU 模式运行"
        return 1
    fi
    print_info "NVIDIA Docker 支持已启用"
    return 0
}

# 构建 Docker 镜像
build() {
    print_info "开始构建 Docker 镜像..."
    docker build -t ${IMAGE_NAME}:${IMAGE_TAG} .
    print_info "镜像构建完成: ${IMAGE_NAME}:${IMAGE_TAG}"
}

# 运行容器（交互式）
run() {
    check_docker
    
    # 检查是否支持 GPU
    GPU_ARGS=""
    if check_nvidia_docker; then
        GPU_ARGS="--gpus all"
    fi
    
    print_info "启动容器: ${CONTAINER_NAME}"
    
    docker run -it --rm \
        ${GPU_ARGS} \
        --name ${CONTAINER_NAME} \
        -v $(pwd):/workspace/llaisys \
        -w /workspace/llaisys \
        ${IMAGE_NAME}:${IMAGE_TAG} \
        /bin/bash
}

# 运行测试
test() {
    check_docker
    
    GPU_ARGS=""
    if check_nvidia_docker; then
        GPU_ARGS="--gpus all"
    fi
    
    print_info "运行测试..."
    
    docker run --rm \
        ${GPU_ARGS} \
        -v $(pwd):/workspace/llaisys \
        -w /workspace/llaisys \
        ${IMAGE_NAME}:${IMAGE_TAG} \
        python3 -m pytest test/
}

# 清理容器和镜像
clean() {
    print_info "清理 Docker 资源..."
    
    # 停止并删除容器
    if docker ps -a | grep -q ${CONTAINER_NAME}; then
        docker stop ${CONTAINER_NAME} 2>/dev/null || true
        docker rm ${CONTAINER_NAME} 2>/dev/null || true
    fi
    
    # 删除镜像
    if docker images | grep -q ${IMAGE_NAME}; then
        docker rmi ${IMAGE_NAME}:${IMAGE_TAG} 2>/dev/null || true
    fi
    
    print_info "清理完成"
}

# 推送镜像到仓库（可选）
push() {
    if [ -z "$1" ]; then
        print_error "请指定仓库地址，例如: ./docker.sh push registry.example.com/llaisys"
        exit 1
    fi
    
    REGISTRY=$1
    print_info "推送镜像到 ${REGISTRY}..."
    
    docker tag ${IMAGE_NAME}:${IMAGE_TAG} ${REGISTRY}:${IMAGE_TAG}
    docker push ${REGISTRY}:${IMAGE_TAG}
    
    print_info "推送完成"
}

# 显示帮助信息
show_help() {
    cat << EOF
llaisys Docker 管理脚本

用法:
    ./docker.sh [命令]

命令:
    build       构建 Docker 镜像
    run         运行交互式容器（挂载当前目录）
    test        在容器中运行测试
    clean       清理容器和镜像
    push <url>  推送镜像到指定仓库
    help        显示此帮助信息

示例:
    ./docker.sh build                    # 构建镜像
    ./docker.sh run                       # 启动交互式容器
    ./docker.sh test                      # 运行测试
    ./docker.sh push registry.com/llaisys # 推送到仓库
    ./docker.sh clean                     # 清理资源

EOF
}

# 主函数
main() {
    case "$1" in
        build)
            check_docker
            build
            ;;
        run)
            run
            ;;
        test)
            test
            ;;
        clean)
            clean
            ;;
        push)
            push "$2"
            ;;
        help|--help|-h|"")
            show_help
            ;;
        *)
            print_error "未知命令: $1"
            show_help
            exit 1
            ;;
    esac
}

main "$@"
