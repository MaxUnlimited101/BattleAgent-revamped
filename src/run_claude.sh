#!/bin/bash

# Use PYTHON env var or fall back to python3 in PATH
PYTHON="${PYTHON:-python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT_PATH="$SCRIPT_DIR/simulation_controller.py"

conflict_names=("Agincourt" "Falkirk" "Poitiers")

for conflict in "${conflict_names[@]}"; do
    echo "Starting simulations for $conflict"
    for ((i=1; i<=5; i++)); do
        "$PYTHON" "$PYTHON_SCRIPT_PATH" \
                    --conflict_name $conflict \
                    --LLM_MODEL "claude" \
                    --is_GPT4V_activate 0 \
                    --simulation_time 90 \
                    --update_interval 15 &
        sleep 5
    done
    wait
    echo "Completed simulations for $conflict"
done
