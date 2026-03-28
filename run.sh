#!/bin/bash

LOG_FILE="logs/run.log"

# Create logs folder
mkdir -p logs

# Overwrite file every run
: > $LOG_FILE

echo "Starting Federated Learning Experiments" | tee -a $LOG_FILE

export PYTHONPATH=$(pwd)

trap "echo 'Cleaning up...' | tee -a $LOG_FILE; pkill -f fl_server; pkill -f run_client; exit" SIGINT

EXPERIMENTS=("baseline" "label_flip" "targeted_flip" "feature_poison" "sign_flip" "scaling" "dp_only" "full_system")

for EXP in "${EXPERIMENTS[@]}"
do
    echo "======================================" | tee -a $LOG_FILE
    echo -e "\n\n==================== $EXP ====================\n" | tee -a $LOG_FILE
    echo "======================================" | tee -a $LOG_FILE

    # Start server
    python -m server.fl_server $EXP >> $LOG_FILE 2>&1 &
    SERVER_PID=$!

    sleep 10

    # Start clients
    CLIENT_PIDS=()
    for i in {1..10}
    do
        python -m clients.run_client $i $EXP >> $LOG_FILE 2>&1 &
        CLIENT_PIDS+=($!)
    done

    sleep 240

    kill $SERVER_PID

    for PID in "${CLIENT_PIDS[@]}"
    do
        kill $PID
    done

    echo "Finished experiment: $EXP" | tee -a $LOG_FILE
    sleep 5
done

echo "All experiments completed" | tee -a $LOG_FILE