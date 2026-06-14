#!/bin/bash

# Use PYTHON env var or fall back to python3 in PATH
PYTHON="${PYTHON:-python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT_PATH="$SCRIPT_DIR/simulation_controller.py"

conflict_names=("Agincourt" "Agincourt" "Agincourt")
LLM_MODELS=("gpt" "gpt" "claude")
is_GPT4V_activates=(0 1 0)
simulation_times=(90 150 90)
update_intervals=(15 15 15)

for j in {0..2}; do
  for i in {1..5}; do
    "$PYTHON" "$PYTHON_SCRIPT_PATH" \
                    --conflict_name ${conflict_names[$j]} \
                    --LLM_MODEL ${LLM_MODELS[$j]} \
                    --is_GPT4V_activate ${is_GPT4V_activates[$j]} \
                    --simulation_time ${simulation_times[$j]} \
                    --update_interval ${update_intervals[$j]} &
    sleep 5
  done
done
wait
