#!/bin/bash
# Runs LOCALLY (nohup'd + disowned as a real OS process at launch time --
# see the invocation used to start it) -- independent of the Claude Code
# process tree, so it survives the Claude session/context ending. Depends
# on this laptop staying powered on and networked; if it's asleep at the
# deadline the pull just doesn't happen automatically, but nothing is lost
# -- the companion vm_timed_stop.sh script (running on the VM itself) stops
# the VM on its own schedule regardless, and the data stays on the VM's
# persistent disk for a manual pull later.
#
# Usage:
#   nohup bash local_timed_pull.sh <hours> <instance> <zone> <project> <run_name> <dest_dir> \
#       > /path/to/local_timed_pull.log 2>&1 < /dev/null &
#   disown
set -uo pipefail

HOURS="${1:?hours required}"
INSTANCE="${2:?instance required}"
ZONE="${3:?zone required}"
PROJECT="${4:?project required}"
RUN_NAME="${5:?run_name required}"
DEST="${6:?dest_dir required}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

# Small buffer past the VM's own deadline so its checkpoints/videos are
# already settled by the time we pull.
SECONDS_FROM_HOURS=$(awk "BEGIN { printf \"%d\", $HOURS * 3600 }")
DEADLINE=$(( $(date +%s) + SECONDS_FROM_HOURS + 300 ))
log "local_timed_pull.sh started. hours=$HOURS deadline=$(date -r "$DEADLINE" 2>/dev/null || date -d @"$DEADLINE")"

while [ "$(date +%s)" -lt "$DEADLINE" ]; do
    sleep 60
done

log "Deadline reached. Checking VM state..."
STATE=$(gcloud compute instances describe "$INSTANCE" --zone="$ZONE" --project="$PROJECT" --format='value(status)' 2>>/dev/stderr || echo "UNKNOWN")
log "VM state: $STATE"

if [ "$STATE" = "RUNNING" ]; then
    log "Pulling all available checkpoint/video pairs for $RUN_NAME..."
    cd "$SCRIPT_DIR"
    bash move_garbage/scripts/pull_run_artifacts.sh "$RUN_NAME" --dest "$DEST"
    PULL_EXIT=$?
    log "Pull finished, exit code $PULL_EXIT."

    log "Signaling the VM that the local pull is complete..."
    gcloud compute ssh "$INSTANCE" --zone="$ZONE" --project="$PROJECT" --tunnel-through-iap \
        --command="touch /tmp/LOCAL_PULL_COMPLETE" || true

    log "Requesting VM stop (idempotent -- harmless if the VM already self-stopped around the same time)..."
    gcloud compute instances stop "$INSTANCE" --zone="$ZONE" --project="$PROJECT" || true
else
    log "VM is not RUNNING (state=$STATE) -- it has likely already self-stopped via vm_timed_stop.sh. Not auto-restarting it just to pull data (would incur extra cost); data remains safe on the persistent disk. Pull manually later: gcloud compute instances start $INSTANCE --zone=$ZONE --project=$PROJECT, then re-run this script's pull step by hand."
fi

log "Done."
