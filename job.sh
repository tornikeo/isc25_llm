#!/bin/bash
#SBATCH --job-name=llama
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8

head_node_ip=10.0.1.22

export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0   # Change to your interface if needed
export NCCL_IB_DISABLE=1

nvidia-smi -pl 400
echo "Running on host:: $head_node_ip"
export LOGLEVEL=INFO 
module list

torchrun \
  --nnodes 1 \
  --nproc_per_node 4 \
  --rdzv_id 424242 \
  --rdzv_backend c10d \
  --rdzv_endpoint $head_node_ip:29500 \
  main.py --benchmark speed --device-type cuda