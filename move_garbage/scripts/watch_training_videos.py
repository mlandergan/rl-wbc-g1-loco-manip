#!/usr/bin/env python3
"""Render rollout videos whenever a matching training checkpoint appears."""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path


CHECKPOINT_PATTERN = re.compile(r"model_step_(\d+)\.pt$")
STOP_REQUESTED = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--interval", type=int, default=500)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--stable-seconds", type=float, default=5.0)
    parser.add_argument("--play-steps", type=int, default=800)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--keep-trajectories", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be positive")
    if args.poll_seconds < 0 or args.stable_seconds < 0:
        parser.error("poll and stability durations cannot be negative")
    return args


def checkpoint_step(path: Path) -> int | None:
    match = CHECKPOINT_PATTERN.fullmatch(path.name)
    return int(match.group(1)) if match else None


def is_stable(path: Path, wait_seconds: float) -> bool:
    try:
        before = path.stat()
        if wait_seconds:
            time.sleep(wait_seconds)
        after = path.stat()
    except FileNotFoundError:
        return False
    return (
        before.st_size > 0
        and before.st_size == after.st_size
        and before.st_mtime_ns == after.st_mtime_ns
    )


def load_state(path: Path) -> dict:
    if not path.exists():
        return {"completed": {}, "failed": {}}
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise RuntimeError(f"Could not read watcher state {path}: {error}") from error
    state.setdefault("completed", {})
    state.setdefault("failed", {})
    return state


def save_state(path: Path, state: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_logged(command: list[str], log_path: Path, cwd: Path) -> None:
    print("+", " ".join(command), flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        result = subprocess.run(
            command,
            cwd=cwd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if result.returncode:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}; see {log_path}"
        )


def render_checkpoint(
    checkpoint: Path,
    output_dir: Path,
    repo_root: Path,
    args: argparse.Namespace,
) -> Path:
    step = checkpoint_step(checkpoint)
    if step is None:
        raise ValueError(f"Unsupported checkpoint name: {checkpoint.name}")

    stem = f"model_step_{step:06d}"
    trajectory = output_dir / "trajectories" / f"{stem}.npz"
    final_video = output_dir / f"{stem}.mp4"
    partial_video = output_dir / f".{stem}.partial.mp4"
    log_dir = output_dir / "logs"
    trajectory.parent.mkdir(parents=True, exist_ok=True)

    rollout_command = [
        args.python,
        "-u",
        str(repo_root / "move_garbage/scripts/play_bin_push.py"),
        f"+checkpoint={checkpoint}",
        "++num_envs=1",
        "++headless=True",
        f"++play_steps={args.play_steps}",
        f"++traj_out={trajectory}",
    ]
    replay_command = [
        args.python,
        "-u",
        str(repo_root / "move_garbage/scripts/replay_traj.py"),
        "--headless",
        "--enable_cameras",
        "--traj",
        str(trajectory),
        "--out",
        str(partial_video),
        "--fps",
        str(args.fps),
        "--stride",
        str(args.stride),
        "--force_exit_after_write",
    ]

    if args.dry_run:
        print("+", " ".join(rollout_command))
        print("+", " ".join(replay_command))
        return final_video

    run_logged(rollout_command, log_dir / f"{stem}_rollout.log", repo_root)
    run_logged(replay_command, log_dir / f"{stem}_replay.log", repo_root)
    if not partial_video.is_file() or partial_video.stat().st_size == 0:
        raise RuntimeError(f"Replay did not produce a video: {partial_video}")
    partial_video.replace(final_video)
    if not args.keep_trajectories:
        trajectory.unlink(missing_ok=True)
    return final_video


def request_stop(signum: int, _frame: object) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(f"Received signal {signum}; stopping after the current checkpoint.", flush=True)


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"Run directory does not exist: {run_dir}")
    if not (run_dir / "config.yaml").is_file():
        raise SystemExit(f"Training config is missing: {run_dir / 'config.yaml'}")

    repo_root = Path(__file__).resolve().parents[2]
    output_dir = (args.output_dir or run_dir / "videos_every_500").expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "watcher_state.json"
    state = load_state(state_path)

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    print(
        f"Watching {run_dir} for every {args.interval} iterations; videos -> {output_dir}",
        flush=True,
    )

    while not STOP_REQUESTED:
        candidates = []
        for checkpoint in run_dir.glob("model_step_*.pt"):
            step = checkpoint_step(checkpoint)
            if step is not None and step % args.interval == 0:
                candidates.append((step, checkpoint.resolve()))

        for step, checkpoint in sorted(candidates):
            key = str(step)
            if key in state["completed"]:
                continue
            if key in state["failed"] and not args.retry_failed:
                continue
            if not is_stable(checkpoint, args.stable_seconds):
                print(f"Checkpoint is still changing; deferring {checkpoint.name}", flush=True)
                continue

            print(f"Rendering iteration {step} from {checkpoint}", flush=True)
            try:
                video = render_checkpoint(checkpoint, output_dir, repo_root, args)
            except Exception as error:
                print(f"Video generation failed for iteration {step}: {error}", flush=True)
                state["failed"][key] = {
                    "checkpoint": str(checkpoint),
                    "error": str(error),
                    "time": time.time(),
                }
                save_state(state_path, state)
            else:
                if not args.dry_run:
                    state["completed"][key] = {
                        "checkpoint": str(checkpoint),
                        "video": str(video),
                        "time": time.time(),
                    }
                    state["failed"].pop(key, None)
                    save_state(state_path, state)
                print(f"Video ready: {video}", flush=True)

            if STOP_REQUESTED:
                break

        if args.once:
            break
        time.sleep(args.poll_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
