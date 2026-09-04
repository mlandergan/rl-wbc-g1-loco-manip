#!/bin/bash
# Runs ON the GCP VM (nohup'd, detached from the launching SSH session) --
# NOT dependent on the Claude Code session, the local laptop, or the SSH
# connection that launched it staying alive. Waits for a deadline (or the
# training tmux session dying early), stops training + the video watcher
# gracefully, gives a grace window for the local machine to pull data, then
# self-shuts-down the VM via `sudo shutdown -h now` -- GCE treats a
# guest-initiated shutdown the same as `gcloud compute instances stop`:
# the persistent disk is NOT deleted, the instance just goes to TERMINATED.
#
# Usage (on the VM):
#   nohup bash vm_timed_stop.sh <hours> <train_tmux_session> <watcher_tmux_session> \
#       > /tmp/vm_timed_stop.log 2>&1 < /dev/null &
#   disown
set -uo pipefail

HOURS="${1:-4}"
TRAIN_TMUX="${2:-finger_control_train}"
WATCHER_TMUX="${3:-video_watcher}"
WATCHER_GRACE_SECONDS=360      # let the last in-flight video render finish
LOCAL_PULL_GRACE_SECONDS=1200  # 20 min for the local machine to wake up and pull

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

SECONDS_FROM_HOURS=$(awk "BEGIN { printf \"%d\", $HOURS * 3600 }")
DEADLINE=$(( $(date +%s) + SECONDS_FROM_HOURS ))
log "vm_timed_stop.sh started. hours=$HOURS deadline=$(date -d @"$DEADLINE" 2>/dev/null || date -r "$DEADLINE")"

# Wait until the deadline, or exit early if the training tmux session dies
# on its own (crash, or it happened to finish its own num_total_batches).
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
    if ! tmux has-session -t "$TRAIN_TMUX" 2>/dev/null; then
        log "Training tmux session '$TRAIN_TMUX' is gone (exited or crashed) -- proceeding early."
        break
    fi
    sleep 30
done

log "Deadline reached (or training already stopped). Stopping training gracefully..."
tmux send-keys -t "$TRAIN_TMUX" C-c 2>/dev/null || true
sleep 5
tmux kill-session -t "$TRAIN_TMUX" 2>/dev/null || true
log "Training tmux session stopped (or was already gone)."

log "Waiting ${WATCHER_GRACE_SECONDS}s for the video watcher to finish any in-flight render..."
sleep "$WATCHER_GRACE_SECONDS"

log "Stopping video watcher..."
tmux kill-session -t "$WATCHER_TMUX" 2>/dev/null || true

log "Writing done-marker for the local puller."
touch /tmp/RUN_DONE_READY_FOR_PULL

log "Waiting up to ${LOCAL_PULL_GRACE_SECONDS}s for local machine to confirm its pull (touch /tmp/LOCAL_PULL_COMPLETE)..."
GRACE_DEADLINE=$(( $(date +%s) + LOCAL_PULL_GRACE_SECONDS ))
while [ "$(date +%s)" -lt "$GRACE_DEADLINE" ]; do
    if [ -f /tmp/LOCAL_PULL_COMPLETE ]; then
        log "Local pull confirmed complete."
        break
    fi
    sleep 30
done
if [ ! -f /tmp/LOCAL_PULL_COMPLETE ]; then
    log "No local-pull confirmation received within the grace window -- proceeding anyway. Data is safe on the persistent disk regardless (VM is being STOPPED, not deleted)."
fi

log "Self-shutting-down the VM now. Disk persists; this is equivalent to 'gcloud compute instances stop', not delete."
sudo shutdown -h now
