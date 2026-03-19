#!/bin/bash

echo "Starting Federated Learning Experiment"

GPU_ID=2

export CUDA_VISIBLE_DEVICES=$GPU_ID

echo "Using GPU: $GPU_ID"

# add project root to python path
export PYTHONPATH=$(pwd)

python experiments/run_baseline_fl.py