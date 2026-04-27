#!/bin/bash

LOG_FILE="logs/run.log"
mkdir -p logs
: > $LOG_FILE
echo "Starting Federated Learning Experiments" | tee -a $LOG_FILE

export PYTHONPATH=$(pwd)
trap "echo 'Cleaning up...' | tee -a $LOG_FILE; pkill -f fl_server; pkill -f run_client; exit" SIGINT

EXPERIMENTS=(
"baseline"

#attack only
"label_flip_only"
"targeted_flip_only"
"feature_poison_only"
"sign_flip_only"

#attack + defense
"label_flip_median" "label_flip_trimmed" "label_flip_krum"
"targeted_flip_median" "targeted_flip_trimmed" "targeted_flip_krum"
"feature_poison_median" "feature_poison_trimmed" "feature_poison_krum" 
"sign_flip_median" "sign_flip_trimmed" "sign_flip_krum"

#DP experiments
"dp_local_eps1" 
"dp_local_eps2" 
"dp_local_eps5" 
"dp_local_adaptive" 

# final system (TAP-FL)
"final_system"
"final_system_targeted" 
"final_system_feature" 
"final_system_sign"
)

for EXP in "${EXPERIMENTS[@]}"
do
    echo -e "\n\n$EXP\n" | tee -a $LOG_FILE
    #start server
    python -m server.fl_server $EXP $DATASET >> $LOG_FILE 2>&1 &
    SERVER_PID=$!
    echo "Waiting for server to be ready..." | tee -a $LOG_FILE
    while ! nc -z 127.0.0.1 8081; do
        sleep 1
    done
    echo "Server is ready!" | tee -a $LOG_FILE
    CLIENT_PIDS=()
    GPU_LIST=(0 3 4 5)   #choose available cuda from nvidia-smi
    NUM_GPUS=${#GPU_LIST[@]}
    for i in {1..10}
    do
        GPU_ID=${GPU_LIST[$(( (i-1) % NUM_GPUS ))]}
        echo "Client $i → GPU $GPU_ID" | tee -a $LOG_FILE
        CUDA_VISIBLE_DEVICES=$GPU_ID python -m clients.run_client $i $EXP $DATASET  >> $LOG_FILE 2>&1 &
        CLIENT_PIDS+=($!)
    done
    wait $SERVER_PID
    for PID in "${CLIENT_PIDS[@]}"
    do
        kill $PID
    done
    echo "Finished experiment: $EXP" | tee -a $LOG_FILE
    sleep 5
done
echo "All experiments completed" | tee -a $LOG_FILE