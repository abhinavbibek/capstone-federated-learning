#!/bin/bash

echo "Starting Federated Learning Experiments"

export PYTHONPATH=$(pwd)

EXPERIMENTS=("baseline" "attack_only" "dp_only" "full_system")

for EXP in "${EXPERIMENTS[@]}"
do
    echo "======================================"
    echo "Running experiment: $EXP"
    echo "======================================"

    # Start server
    python -m server.fl_server $EXP &
    SERVER_PID=$!

    sleep 5

    # Start clients
    for i in {1..5}
    do
        python -m clients.run_client $i $EXP &
    done

    # Wait for training to complete
    sleep 120

    # Kill processes cleanly
    kill $SERVER_PID
    pkill -f run_client

    echo "Finished experiment: $EXP"
    sleep 5
done

echo "All experiments completed"