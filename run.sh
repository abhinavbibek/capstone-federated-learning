#!/bin/bash

echo "Starting Federated Learning Experiments"

export PYTHONPATH=$(pwd)

trap "echo 'Cleaning up...'; pkill -f fl_server; pkill -f run_client; exit" SIGINT

EXPERIMENTS=("baseline" "attack_only" "dp_only" "full_system")

for EXP in "${EXPERIMENTS[@]}"
do
    echo "======================================"
    echo "Running experiment: $EXP"
    echo "======================================"

    # Start server
    python -m server.fl_server $EXP &
    SERVER_PID=$!

    sleep 10   # ✅ increased

    # Start clients
    CLIENT_PIDS=()
    for i in {1..10}
    do
        python -m clients.run_client $i $EXP &
        CLIENT_PIDS+=($!)
    done

    # Wait for training
    sleep 240   # safer

    # Kill processes cleanly
    kill $SERVER_PID

    for PID in "${CLIENT_PIDS[@]}"
    do
        kill $PID
    done

    echo "Finished experiment: $EXP"
    sleep 5
done

echo "All experiments completed"