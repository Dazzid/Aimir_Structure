#!/bin/bash
# Launch musical_form_analysis.py inside the Docker container via tmux.
# Creates a tmux session called "aimir".
#
# From the host:
#   bash launch_musical_form.sh
#
# To attach:
#   tmux attach -t aimir

SESSION="aimir"
CONTAINER="dalmazzo_aimir_structure"

tmux kill-session -t "$SESSION" 2>/dev/null

tmux new-session -d -s "$SESSION" \
  "docker exec -it $CONTAINER python3.11 /workspace/src/musical_form_analysis.py --n_jobs 8 2>&1 | tee /workspace/musical_form_log.txt; echo 'DONE — press Enter to close'; read"

echo "tmux session '$SESSION' created."
echo "Attach with:  tmux attach -t $SESSION"
