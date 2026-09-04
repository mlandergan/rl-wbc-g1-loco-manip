#!/usr/bin/env bash
# Pull (checkpoint, video) pairs for a training run from the GCP VM into a
# local, per-run, per-step folder. Reads videos_every_*/watcher_state.json
# (written by watch_training_videos.py) as the source of truth for which
# videos have actually finished rendering -- never pulls a partial video.
# Safe to re-run: already-pulled steps are skipped.
set -euo pipefail

RUN_NAME="${1:?Usage: pull_run_artifacts.sh <run_name> [--vm NAME] [--zone ZONE] [--project PROJECT] [--dest DIR]}"
shift

VM="g1-bin-push-l4"
ZONE="us-east1-b"
PROJECT="g1-rl-training"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEST="${REPO_ROOT}/runs/${RUN_NAME}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --vm) VM="$2"; shift 2 ;;
    --zone) ZONE="$2"; shift 2 ;;
    --project) PROJECT="$2"; shift 2 ;;
    --dest) DEST="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

REMOTE_RUN_DIR="~/move_garbage_open_source/logs_rl/g1_bin_push_homie/${RUN_NAME}"
SSH_PREFIX=(gcloud compute ssh "$VM" --zone="$ZONE" --project="$PROJECT" --tunnel-through-iap)
SCP_PREFIX=(gcloud compute scp --zone="$ZONE" --project="$PROJECT" --tunnel-through-iap)

mkdir -p "$DEST"
STATE_LOCAL="$(mktemp)"
trap 'rm -f "$STATE_LOCAL"' EXIT

# There's exactly one videos_every_* dir per run (named after whatever
# --interval watch_training_videos.py was launched with).
VIDEOS_DIR_NAME="$("${SSH_PREFIX[@]}" --command="ls -d ${REMOTE_RUN_DIR}/videos_every_* 2>/dev/null | xargs -n1 basename | head -1" 2>/dev/null | tail -1)"
if [[ -z "$VIDEOS_DIR_NAME" ]]; then
  echo "No videos_every_* directory found under ${REMOTE_RUN_DIR} yet." >&2
  echo "Is watch_training_videos.py running for this run? Nothing to pull." >&2
  exit 1
fi
REMOTE_VIDEOS_DIR="${REMOTE_RUN_DIR}/${VIDEOS_DIR_NAME}"

"${SCP_PREFIX[@]}" "${VM}:${REMOTE_VIDEOS_DIR}/watcher_state.json" "$STATE_LOCAL" >/dev/null 2>&1 || {
  echo "Could not fetch watcher_state.json from ${REMOTE_VIDEOS_DIR} -- has anything rendered yet?" >&2
  exit 1
}

STEPS="$(python3 -c "
import json
with open('${STATE_LOCAL}') as f:
    state = json.load(f)
for step in sorted(int(s) for s in state.get('completed', {})):
    print(step)
")"

if [[ -z "$STEPS" ]]; then
  echo "watcher_state.json has no completed renders yet for ${RUN_NAME}."
  exit 0
fi

PULLED=0
SKIPPED=0
for STEP in $STEPS; do
  STEM="$(printf 'model_step_%06d' "$STEP")"
  STEP_DEST="${DEST}/step_$(printf '%06d' "$STEP")"
  if [[ -f "${STEP_DEST}/${STEM}.pt" && -f "${STEP_DEST}/${STEM}.mp4" ]]; then
    SKIPPED=$((SKIPPED + 1))
    continue
  fi
  mkdir -p "$STEP_DEST"
  echo "Pulling ${STEM} -> ${STEP_DEST}"
  "${SCP_PREFIX[@]}" "${VM}:${REMOTE_RUN_DIR}/${STEM}.pt" "${STEP_DEST}/${STEM}.pt"
  "${SCP_PREFIX[@]}" "${VM}:${REMOTE_VIDEOS_DIR}/${STEM}.mp4" "${STEP_DEST}/${STEM}.mp4"
  PULLED=$((PULLED + 1))
done

echo "Done. Pulled ${PULLED} new (checkpoint, video) pair(s), ${SKIPPED} already present."
echo "Local artifacts: ${DEST}/step_NNNNNN/{${RUN_NAME##*/}... }"
