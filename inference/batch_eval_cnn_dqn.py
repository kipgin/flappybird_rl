import argparse
import csv
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

# Ensure repo root is on sys.path so we can import inference.infer_cnn
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from inference.infer_cnn import (  # noqa: E402
    build_model,
    default_config_key_for_algo,
    load_cfg,
    load_checkpoint_into_model,
    make_env,
    obs_to_tensor_single,
    select_action,
    select_device,
)


_CKPT_TS_RE = re.compile(r"best_model_(\d{8}_\d{6})\.pth$")
_TS_RE = re.compile(r"(\d{8}_\d{6})")


@dataclass
class EvalResult:
    ckpt_time: str
    epoch: int | None
    avg_reward: float
    loss: float | None
    eval_seconds: float
    model_path: str


def parse_checkpoint_time(model_path: str) -> str | None:
    m = _CKPT_TS_RE.search(os.path.basename(model_path))
    return m.group(1) if m else None


def normalize_timestamp(value: str | None) -> str | None:
    if not value:
        return None
    m = _TS_RE.search(value)
    return m.group(1) if m else None


def find_checkpoint_by_key(checkpoints: list[Path], key: str) -> Path | None:
    key = key.strip()
    if not key:
        return None
    ts = normalize_timestamp(key)

    # Exact filename match first
    for p in checkpoints:
        if p.name == key or p.stem == key:
            return p

    # Timestamp match
    if ts:
        for p in checkpoints:
            if parse_checkpoint_time(str(p)) == ts:
                return p

    # Substring match (fallback)
    for p in checkpoints:
        if key in p.name:
            return p
    return None


def list_checkpoints(checkpoints_dir: Path) -> list[Path]:
    if not checkpoints_dir.exists():
        raise FileNotFoundError(f"Checkpoints dir not found: {checkpoints_dir}")
    ckpts = sorted(
        [p for p in checkpoints_dir.glob("*.pth") if p.is_file()],
        key=lambda p: parse_checkpoint_time(str(p)) or p.name,
    )
    return ckpts


def _iter_run_dirs(training_runs_dir: Path) -> list[Path]:
    if not training_runs_dir.exists():
        return []
    runs = []
    for p in training_runs_dir.iterdir():
        if p.is_dir() and re.fullmatch(r"\d{8}_\d{6}", p.name):
            runs.append(p)
    runs.sort(key=lambda p: p.name)
    return runs


def _pick_nearest_run_dir(training_runs_dir: Path, ckpt_time: str | None) -> Path | None:
    runs = _iter_run_dirs(training_runs_dir)
    if not runs:
        return None
    if not ckpt_time:
        return runs[-1]

    # Prefer latest run <= checkpoint timestamp; else fallback to latest.
    candidates = [r for r in runs if r.name <= ckpt_time]
    return candidates[-1] if candidates else runs[-1]


def find_loss_from_training_logs(
    *,
    training_runs_dir: Path,
    ckpt_time: str | None,
    target_epoch: int | None,
) -> float | None:
    if target_epoch is None:
        return None

    run_dir = _pick_nearest_run_dir(training_runs_dir, ckpt_time)
    if run_dir is None:
        return None

    log_path = run_dir / "train_performance_log.csv"
    if not log_path.exists():
        return None

    try:
        with log_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    if int(float(row.get("epoch", "nan"))) == int(target_epoch):
                        v = row.get("loss")
                        return float(v) if v is not None and v != "" else None
                except Exception:
                    continue
    except Exception:
        return None

    return None


def evaluate_checkpoint(
    *,
    model_path: Path,
    algo: str,
    cfg: dict,
    env_id: str,
    num_frames: int,
    device: torch.device,
    seeds: list[int],
    episodes_per_seed: int,
    max_steps: int,
    deterministic: bool,
) -> tuple[float, float]:
    # Create env once per checkpoint; change seed via reset().
    env = make_env(
        env_id,
        cfg,
        seed=seeds[0] if seeds else 0,
        num_frames=num_frames,
        record_video_dir=None,
        show_window=False,
    )

    obs, _ = env.reset(seed=seeds[0] if seeds else 0)
    obs_shape = tuple(np.array(obs).shape)
    action_dim = env.action_space.n

    model = build_model(algo, cfg, obs_shape=obs_shape, action_dim=action_dim)
    model = load_checkpoint_into_model(model, str(model_path), device)

    # Warm-up
    with torch.no_grad():
        dummy = torch.zeros((1, *obs_shape), device=device, dtype=torch.float32)
        if algo == "cnn_dqn":
            _ = model(dummy)
        else:
            _ = model.get_action_and_value(dummy)

    t0 = time.perf_counter()

    seed_means: list[float] = []
    for seed in seeds:
        env.action_space.seed(seed)
        ep_rewards: list[float] = []
        for ep in range(episodes_per_seed):
            obs, _ = env.reset(seed=seed + ep)
            total_r = 0.0
            for _ in range(max_steps):
                obs_t = obs_to_tensor_single(obs, device)
                action = select_action(model, algo, obs_t, deterministic=deterministic)
                obs, reward, terminated, truncated, _info = env.step(action)
                total_r += float(reward)
                if terminated or truncated:
                    break
            ep_rewards.append(total_r)
        seed_means.append(float(np.mean(ep_rewards)) if ep_rewards else 0.0)

    avg_reward = float(np.mean(seed_means)) if seed_means else 0.0
    elapsed = time.perf_counter() - t0

    env.close()
    return avg_reward, elapsed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", default="cnn_dqn", choices=["cnn_dqn", "cnn_ppo", "cnn_policy_gradient"])
    ap.add_argument(
        "--checkpoints-dir",
        type=str,
        default=str(REPO_ROOT / "training" / "flappy_bird_cnn_dqn_checkpoints"),
    )
    ap.add_argument("--yaml", type=str, default=str(REPO_ROOT / "hyperparameters.yml"))
    ap.add_argument("--config-key", type=str, default=None)
    ap.add_argument("--episodes-per-seed", type=int, default=5)
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4")
    ap.add_argument("--max-steps", type=int, default=5000)
    ap.add_argument("--deterministic", action="store_true", default=True)
    ap.add_argument("--force-cpu", action="store_true")
    ap.add_argument(
        "--from",
        dest="from_key",
        type=str,
        default=None,
        help="Only evaluate checkpoints with timestamp >= this (e.g. 20260106_105434 or best_model_20260106_105434).",
    )
    ap.add_argument(
        "--only",
        dest="only_key",
        type=str,
        default=None,
        help="Evaluate only one checkpoint by name/timestamp (e.g. best_model_20260106_105434.pth).",
    )
    ap.add_argument(
        "--output",
        type=str,
        default=str(REPO_ROOT / "training" / "flappy_bird_cnn_dqn_checkpoints" / "batch_eval_results.csv"),
    )
    args = ap.parse_args()

    device = select_device(force_cpu=args.force_cpu)

    config_key = args.config_key or default_config_key_for_algo(args.algo)
    cfg = load_cfg(args.yaml, config_key)

    env_id = cfg.get("env_id", "FlappyBird-v0")
    num_frames = int(cfg.get("frame_stack", 4))

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip() != ""]

    checkpoints_dir = Path(args.checkpoints_dir)
    ckpts = list_checkpoints(checkpoints_dir)
    if not ckpts:
        raise RuntimeError(f"No .pth checkpoints found in {checkpoints_dir}")

    if args.only_key:
        picked = find_checkpoint_by_key(ckpts, args.only_key)
        if picked is None:
            raise RuntimeError(f"Could not find checkpoint matching: {args.only_key}")
        ckpts = [picked]

    if args.from_key:
        from_ts = normalize_timestamp(args.from_key)
        if from_ts is None:
            raise RuntimeError(f"Could not parse timestamp from --from: {args.from_key}")
        ckpts = [p for p in ckpts if (parse_checkpoint_time(str(p)) or "") >= from_ts]
        if not ckpts:
            raise RuntimeError(f"No checkpoints with timestamp >= {from_ts} in {checkpoints_dir}")

    training_runs_dir = REPO_ROOT / "training" / "flappy_bird_cnn_dqn"

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "epoch", "avg_reward", "loss"])

        for model_path in ckpts:
            ckpt_time = parse_checkpoint_time(str(model_path))
            ckpt = torch.load(model_path, map_location="cpu")
            epoch = None
            if isinstance(ckpt, dict):
                if "epoch" in ckpt:
                    try:
                        epoch = int(ckpt["epoch"])
                    except Exception:
                        epoch = None

            avg_reward, eval_seconds = evaluate_checkpoint(
                model_path=model_path,
                algo=args.algo,
                cfg=cfg,
                env_id=env_id,
                num_frames=num_frames,
                device=device,
                seeds=seeds,
                episodes_per_seed=int(args.episodes_per_seed),
                max_steps=int(args.max_steps),
                deterministic=bool(args.deterministic),
            )

            loss = find_loss_from_training_logs(
                training_runs_dir=training_runs_dir,
                ckpt_time=ckpt_time,
                target_epoch=epoch,
            )

            writer.writerow([
                ckpt_time or "",
                "" if epoch is None else epoch,
                f"{avg_reward:.6f}",
                "" if loss is None else f"{loss:.6f}",
            ])

            print(
                f"{model_path.name}: epoch={epoch} avg_reward={avg_reward:.3f} "
                f"loss={'NA' if loss is None else f'{loss:.6f}'} "
                f"eval_seconds={eval_seconds:.2f}"
            )

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
