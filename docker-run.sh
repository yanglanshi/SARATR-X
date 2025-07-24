#!/bin/bash

# SARATR-X Docker 设置和使用脚本
# 此脚本提供了使用Docker构建和运行SARATR-X的实用工具

set -e

# 输出颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # 无颜色

# 打印彩色输出的函数
print_color() {
    printf "${1}${2}${NC}\n"
}

# 检查是否安装了Docker的函数
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_color $RED "Docker未安装。请先安装Docker。"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        print_color $RED "Docker Compose未安装。请先安装Docker Compose。"
        exit 1
    fi
}

# 检查NVIDIA Docker支持的函数
check_nvidia_docker() {
    if ! docker run --rm --gpus all nvidia/cuda:11.1-base-ubuntu20.04 nvidia-smi &> /dev/null; then
        print_color $YELLOW "警告：未检测到NVIDIA Docker支持。GPU加速可能无法工作。"
        print_color $YELLOW "请安装nvidia-docker2以获得GPU支持。"
    else
        print_color $GREEN "检测到NVIDIA Docker支持。"
    fi
}

# 创建必要目录的函数
create_directories() {
    print_color $BLUE "创建必要的目录..."
    mkdir -p data output checkpoints work_dirs logs
    print_color $GREEN "目录创建成功。"
}

# 构建Docker镜像的函数
build_image() {
    print_color $BLUE "构建SARATR-X Docker镜像..."
    docker-compose build saratrx
    print_color $GREEN "Docker镜像构建成功。"
}

# 启动服务的函数
start_services() {
    print_color $BLUE "启动SARATR-X服务..."
    docker-compose up -d
    print_color $GREEN "服务启动成功。"
    print_color $YELLOW "访问点："
    print_color $YELLOW "  - 主容器: docker-compose exec saratrx bash"
    print_color $YELLOW "  - TensorBoard: http://localhost:6007"
    print_color $YELLOW "  - Jupyter: http://localhost:8889"
}

# 停止服务的函数
stop_services() {
    print_color $BLUE "停止SARATR-X服务..."
    docker-compose down
    print_color $GREEN "服务停止成功。"
}

# 进入主容器的函数
enter_container() {
    print_color $BLUE "进入SARATR-X主容器..."
    docker-compose exec saratrx bash
}

# 运行预训练的函数
run_pretraining() {
    print_color $BLUE "开始预训练..."
    docker-compose exec saratrx bash -c "
        cd /workspace/pre-training && \
        python -m torch.distributed.launch --nproc_per_node=1 --master_port=12345 --use_env main_pretrain.py \
            --data_path /workspace/data \
            --output_dir /workspace/output \
            --model mae_hivit_base_dec512d6b \
            --mask_ratio 0.75 \
            --batch_size 32 \
            --accum_iter 4 \
            --blr 1.5e-4 \
            --weight_decay 0.05 \
            --epochs 100 \
            --warmup_epochs 5
    "
}

# 运行分类任务的函数
run_classification() {
    print_color $BLUE "开始分类任务..."
    docker-compose exec saratrx bash -c "
        cd /workspace/classification && \
        bash MIM_linear.sh
    "
}

# 运行检测任务的函数
run_detection() {
    print_color $BLUE "开始检测任务..."
    docker-compose exec saratrx bash -c "
        cd /workspace/detection && \
        ./tools/dist_train.sh configs/_hivit_/hivit_base_SARDet.py 1 --work-dir /workspace/work_dirs/SARDet
    "
}

# 显示日志的函数
show_logs() {
    docker-compose logs -f saratrx
}

# 清理资源的函数
cleanup() {
    print_color $BLUE "清理Docker资源..."
    docker-compose down -v
    docker system prune -f
    print_color $GREEN "清理完成。"
}

# 显示帮助信息的函数
show_help() {
    echo "SARATR-X Docker 管理脚本"
    echo ""
    echo "用法: $0 [命令]"
    echo ""
    echo "命令："
    echo "  setup           - 检查需求并创建目录"
    echo "  build           - 构建Docker镜像"
    echo "  start           - 启动所有服务"
    echo "  stop            - 停止所有服务"
    echo "  shell           - 进入主容器shell"
    echo "  pretrain        - 运行预训练"
    echo "  classify        - 运行分类任务"
    echo "  detect          - 运行检测任务"
    echo "  logs            - 显示容器日志"
    echo "  cleanup         - 清理Docker资源"
    echo "  help            - 显示此帮助信息"
    echo ""
    echo "示例："
    echo "  $0 setup        # 初始设置"
    echo "  $0 build        # 构建Docker镜像"
    echo "  $0 start        # 启动服务"
    echo "  $0 shell        # 进入容器进行手动操作"
}

# 主脚本逻辑
case "${1:-help}" in
    setup)
        check_docker
        check_nvidia_docker
        create_directories
        ;;
    build)
        check_docker
        build_image
        ;;
    start)
        check_docker
        start_services
        ;;
    stop)
        stop_services
        ;;
    shell)
        enter_container
        ;;
    pretrain)
        run_pretraining
        ;;
    classify)
        run_classification
        ;;
    detect)
        run_detection
        ;;
    logs)
        show_logs
        ;;
    cleanup)
        cleanup
        ;;
    help|*)
        show_help
        ;;
esac