"""Create a resolved RoboCOIN config from an original FastWAM baseline task."""
from __future__ import annotations

import argparse
import json
from math import ceil
import os
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fastwam-repo", type=Path, required=True)
    parser.add_argument("--baseline", choices=("fastwam", "fastwam_joint"), required=True)
    parser.add_argument("--epochs", type=int, choices=(5, 10), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("/cfsdata/chushan/RoboCOIN-downloads"))
    parser.add_argument("--training-output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    baseline_repo = args.fastwam_repo.expanduser().resolve()
    config_dir = baseline_repo / "configs"
    task = f"galaxea_indoor_cleaning_{args.baseline}_3cam224_1e-4"
    if not (config_dir / "task" / f"{task}.yaml").is_file():
        raise FileNotFoundError(f"The original baseline task is missing: {config_dir / 'task' / (task + '.yaml')}")
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name="train", overrides=[f"task={task}"])
    OmegaConf.set_struct(cfg, False)
    data_root = args.data_root.expanduser().resolve()
    run_dir = data_root / "vae-cache" / "_precompute" / "20260921T111155Z"
    rift_root = Path(__file__).resolve().parents[1]
    cfg.data = OmegaConf.load(rift_root / "configs" / "data" / "robocoin_multitask.yaml")
    dataset_dirs = [data_root / "training-views" / "galaxea_mm_v1" / name
                    for name in cfg.data.train.expected_total_episodes]
    cfg.data.train.dataset_dirs = [str(path) for path in dataset_dirs]
    cfg.data.train.pretrained_norm_stats = str(data_root / "stat-norm" / "robocoin_multitask_mm" / "dataset_stats.json")
    cfg.data.train.text_embedding_cache_dir = str(data_root / "t5-cache")
    cfg.vae_cache_dir = str(data_root / "vae-cache")
    cfg.data.train.vae_cache_index_dir = str(run_dir)
    cfg.model._target_ = f"rift.baselines.fastwam.create_{args.baseline}"
    cfg.model.action_dit_pretrained_path = os.environ.get(
        "ACTION_DIT_PRETRAINED_PATH",
        "/data/chushan/rift-assets/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt",
    )
    expected_batch = {"fastwam": (32, 1), "fastwam_joint": (16, 2)}[args.baseline]
    if (int(cfg.batch_size), int(cfg.gradient_accumulation_steps)) != expected_batch:
        raise ValueError(f"Original {args.baseline} BS/GAS differs from the audited {expected_batch}.")
    if str(cfg.mixed_precision) != "bf16":
        raise ValueError("Original baseline task must use bf16.")
    counts = {}
    for path in dataset_dirs:
        info = json.loads((path / "meta" / "info.json").read_text())
        expected_episodes = int(cfg.data.train.expected_total_episodes[path.name])
        if int(info["total_episodes"]) != expected_episodes or int(info["total_frames"]) <= 0:
            raise ValueError(f"Unexpected dataset counts: {path}")
        counts[path.name] = {"episodes": int(info["total_episodes"]), "frames": int(info["total_frames"])}
    total_frames = sum(row["frames"] for row in counts.values())
    steps_per_epoch = ceil(ceil(total_frames / (int(cfg.batch_size) * 8)) / int(cfg.gradient_accumulation_steps))
    cfg.num_epochs = args.epochs
    cfg.max_steps = steps_per_epoch * args.epochs
    cfg.resume = None
    cfg.eval_every = 0
    cfg.wandb.enabled = False
    cfg.wandb.name = f"robocoin_{args.baseline}_indexed_{args.epochs}ep"
    training_output = args.training_output or Path(f"/data/chushan/fastwam-runs/robocoin_{args.baseline}_{args.epochs}ep")
    cfg.output_dir = str(training_output.expanduser().resolve())
    OmegaConf.resolve(cfg)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = OmegaConf.to_yaml(cfg, resolve=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(payload)
    print(json.dumps({"config": str(output), "original_task": task, "datasets": counts,
                      "frames": total_frames, "world_size": 8, "batch_size": int(cfg.batch_size),
                      "gradient_accumulation_steps": int(cfg.gradient_accumulation_steps),
                      "steps_per_epoch": steps_per_epoch, "epochs": args.epochs,
                      "max_steps": int(cfg.max_steps), "resume": None, "eval_every": 0,
                      "wandb_enabled": False, "training_output": cfg.output_dir}, sort_keys=True))


if __name__ == "__main__":
    main()
