#!/bin/bash

LOG_FILE="logs/run.log"

# Create logs folder
mkdir -p logs

# Overwrite file every run
: > $LOG_FILE

echo "Starting Federated Learning Experiments" | tee -a $LOG_FILE

export PYTHONPATH=$(pwd)

trap "echo 'Cleaning up...' | tee -a $LOG_FILE; pkill -f fl_server; pkill -f run_client; exit" SIGINT


EXPERIMENTS=(
"baseline"

# # #attack only
"label_flip_only"
"targeted_flip_only"
"feature_poison_only"
"sign_flip_only"
"scaling_only"

# # #attack + defense
"label_flip_median" "label_flip_trimmed" "label_flip_krum" "label_flip_clip"
"targeted_flip_median" "targeted_flip_trimmed" "targeted_flip_krum" "targeted_flip_clip"
"feature_poison_median" "feature_poison_trimmed" "feature_poison_krum" "feature_poison_clip"
"sign_flip_median" "sign_flip_trimmed" "sign_flip_krum" "sign_flip_clip"
"scaling_median" "scaling_trimmed" "scaling_krum" "scaling_clip"

# # #DP experiments
"dp_local_eps1" "dp_local_eps2" "dp_local_eps5" "dp_local_adaptive" "dp_server_fixed"
"dp_server_adaptive" 


# final system experiment
"final_system"


)



for EXP in "${EXPERIMENTS[@]}"
do
    echo "==========================================================" | tee -a $LOG_FILE
    echo -e "\n\n==================== $EXP ====================\n" | tee -a $LOG_FILE
    echo "==========================================================" | tee -a $LOG_FILE

    # Start server
    python -m server.fl_server $EXP $DATASET >> $LOG_FILE 2>&1 &
    SERVER_PID=$!

    echo "Waiting for server to be ready..." | tee -a $LOG_FILE

    while ! nc -z 127.0.0.1 8081; do
        sleep 1
    done

    echo "Server is ready!" | tee -a $LOG_FILE

    # Start clients
    # CLIENT_PIDS=()
    # for i in {1..10}
    # do
    #     python -m clients.run_client $i $EXP >> $LOG_FILE 2>&1 &
    #     CLIENT_PIDS+=($!)
    # done
    
    CLIENT_PIDS=()

    GPU_LIST=(1 2 4)   # 🔥 manually choose from nvidia-smi
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
echo "Running SHAP comparisons..." | tee -a $LOG_FILE
python -m analysis.compare_shap $DATASET >> $LOG_FILE 2>&1
echo "All experiments completed" | tee -a $LOG_FILE