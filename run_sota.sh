#!/bin/bash

LOG_FILE="logs/run_sota.log"

mkdir -p logs
: > $LOG_FILE

echo "Starting SOTA Comparison Experiments" | tee -a $LOG_FILE

export PYTHONPATH=$(pwd)

trap "echo 'Cleaning up...' | tee -a $LOG_FILE; pkill -f fl_server; pkill -f run_client; exit" SIGINT

# =============================
# EXPERIMENTS (SOTA ONLY)
# =============================
EXPERIMENTS=(
"fedprox"
"label_flip_krum"
"fltrust"
)

# =============================
# DATASETS
# =============================
DATASETS=("adult" "credit")

# =============================
# GPU CONFIG
# =============================
GPU_LIST=(1 4 7)
NUM_GPUS=${#GPU_LIST[@]}

# =============================
# MAIN LOOP
# =============================
for DATASET in "${DATASETS[@]}"
do
    echo -e "\n\n==================== DATASET: $DATASET ====================\n" | tee -a $LOG_FILE

    for EXP in "${EXPERIMENTS[@]}"
    do
        echo "==========================================================" | tee -a $LOG_FILE
        echo -e "\n\n==================== $EXP ($DATASET) ====================\n" | tee -a $LOG_FILE
        echo "==========================================================" | tee -a $LOG_FILE

        # =============================
        # START SERVER
        # =============================
        python -m server.fl_server $EXP $DATASET >> $LOG_FILE 2>&1 &
        SERVER_PID=$!

        echo "Waiting for server to be ready..." | tee -a $LOG_FILE

        while ! nc -z 127.0.0.1 8081; do
            sleep 1
        done

        echo "Server is ready!" | tee -a $LOG_FILE

        # =============================
        # START CLIENTS
        # =============================
        CLIENT_PIDS=()

        for i in {1..10}
        do
            GPU_ID=${GPU_LIST[$(( (i-1) % NUM_GPUS ))]}

            echo "Client $i → GPU $GPU_ID" | tee -a $LOG_FILE

            CUDA_VISIBLE_DEVICES=$GPU_ID python -m clients.run_client $i $EXP $DATASET >> $LOG_FILE 2>&1 &

            CLIENT_PIDS+=($!)
        done

        # =============================
        # WAIT FOR SERVER
        # =============================
        wait $SERVER_PID

        # =============================
        # CLEAN CLIENTS
        # =============================
        for PID in "${CLIENT_PIDS[@]}"
        do
            kill $PID 2>/dev/null
        done

        echo "Finished experiment: $EXP ($DATASET)" | tee -a $LOG_FILE

        sleep 5
    done
done

echo "All SOTA experiments completed" | tee -a $LOG_FILE