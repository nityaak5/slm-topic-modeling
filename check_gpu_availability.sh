#!/bin/bash
# Script to check GPU availability and show which GPUs are free

echo "Checking GPU availability across nodes..."
echo "========================================"

# Check GPUs on current node
echo ""
echo "Current node: $(hostname)"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader | \
  awk -F', ' '{
    gpu=$1; name=$2; used=$3; total=$4; util=$5
    gsub(/ MiB/, "", used)
    gsub(/ MiB/, "", total)
    gsub(/ %/, "", util)
    if (used < 1000 && util < 5) {
      status="🟢 FREE"
    } else if (used < total*0.5) {
      status="🟡 PARTIAL"
    } else {
      status="🔴 BUSY"
    }
    printf "GPU %s (%s): %s - %s/%s MB used, %s%% util\n", gpu, name, status, used, total, util
  }'

echo ""
echo "To request a specific GPU, use:"
echo "  srun --gres=gpu:1 --time=02:00:00 --mem=16G --pty bash"
echo ""
echo "Then inside the session, check which GPU you got:"
echo "  echo \$CUDA_VISIBLE_DEVICES"
echo "  nvidia-smi"
