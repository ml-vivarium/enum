import argparse
import os
import shlex
import subprocess
import sys
import time
from multiprocessing import Process
from pathlib import Path


DEFAULT_RULES = (
    "0,1,2,3,4,5,6,7,8,9,10,11,14,15,18,19,22,23,"
    "24,25,26,27,28,29,30,32,33,34,35,36,37,38,39,40,"
    "41,42,43,44,45,46,50,51,54,55,56,57,58,59,60,61,"
    "62,63,72,73,74,75,76,77,78,79,90,91,94,95,104,"
    "105,108,109,110,122,123,126,127,128,129,130,132,"
    "134,138,146,147,150,152,156,160,162,164,170"
)


def parse_rules(value):
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def visible_gpu_ids(requested_gpus):
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible:
        ids = [part.strip() for part in cuda_visible.split(",") if part.strip()]
    else:
        ids = [str(idx) for idx in range(requested_gpus)]
    return ids[:requested_gpus]


def run_command(command, log_path, env):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n$ {shlex.join(command)}\n")
        log.flush()
        return subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env).returncode


def train_rule(args, rule, gpu_id):
    rule_dir = Path(args.output_root) / f"rule_{rule:03d}"
    checkpoint_dir = rule_dir / "checkpoints"
    attention_dir = rule_dir / "attention"
    rule_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    attention_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONUNBUFFERED"] = "1"

    final_checkpoint = checkpoint_dir / "final.pt"
    best_checkpoint = checkpoint_dir / "best.pt"
    train_log = rule_dir / "train.log"
    attention_log = rule_dir / "attention.log"

    common = [
        args.python,
        args.script,
        "--task",
        "lm",
        "--lm-direction",
        "reverse",
        "--mask-row-prefix",
        "--position-encoding",
        "grid",
        "--rule",
        str(rule),
        "--width",
        str(args.width),
        "--frames",
        str(args.frames),
        "--layers",
        str(args.layers),
        "--d-model",
        str(args.d_model),
        "--heads",
        str(args.heads),
        "--batch-size",
        str(args.batch_size),
        "--eval-every",
        str(args.eval_every),
        "--eval-batches",
        str(args.eval_batches),
        "--lr",
        str(args.lr),
        "--dtype",
        args.dtype,
        "--seed",
        str(args.seed + rule),
    ]

    train_command = common + [
        "--steps",
        str(args.steps),
        "--checkpoint-every",
        str(args.checkpoint_every),
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--checkpoint-out",
        str(final_checkpoint),
        "--best-checkpoint-out",
        str(best_checkpoint),
        "--loss-curve-image",
        str(rule_dir / "loss.png"),
    ]
    if args.compile:
        train_command.append("--compile")

    started = time.time()
    print(f"gpu={gpu_id} rule={rule} phase=train start", flush=True)
    code = run_command(train_command, train_log, env)
    elapsed = time.time() - started
    if code != 0:
        print(f"gpu={gpu_id} rule={rule} phase=train failed code={code} elapsed={elapsed:.1f}s", flush=True)
        return code
    print(f"gpu={gpu_id} rule={rule} phase=train done elapsed={elapsed:.1f}s", flush=True)

    attention_command = common + [
        "--steps",
        "0",
        "--checkpoint-in",
        str(best_checkpoint),
        "--attention-image",
        str(attention_dir / "attention.png"),
        "--attention-targets",
        args.attention_target,
        "--attention-all-heads",
        "--attention-order",
        "natural",
        "--attention-overlay",
        "--evolution-image",
        str(attention_dir / "evolution.png"),
    ]
    print(f"gpu={gpu_id} rule={rule} phase=attention start", flush=True)
    code = run_command(attention_command, attention_log, env)
    if code != 0:
        print(f"gpu={gpu_id} rule={rule} phase=attention failed code={code}", flush=True)
        return code
    print(f"gpu={gpu_id} rule={rule} phase=attention done", flush=True)
    return 0


def worker(args, gpu_id, assigned_rules):
    print(f"gpu={gpu_id} assigned_rules={','.join(map(str, assigned_rules))}", flush=True)
    failed = []
    for rule in assigned_rules:
        code = train_rule(args, rule, gpu_id)
        if code != 0:
            failed.append(rule)
    if failed:
        print(f"gpu={gpu_id} failed_rules={','.join(map(str, failed))}", flush=True)
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rules", default=DEFAULT_RULES)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--script", default="rule110_tiny_transformer.py")
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--dtype", choices=("fp32", "bf16", "fp16"), default="bf16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--attention-target", default="16:16")
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    rules = parse_rules(args.rules)
    gpu_ids = visible_gpu_ids(args.gpus)
    if not gpu_ids:
        raise SystemExit("no GPUs available")

    assignments = [rules[idx::len(gpu_ids)] for idx in range(len(gpu_ids))]
    Path(args.output_root).mkdir(parents=True, exist_ok=True)
    print(
        f"rules={len(rules)} output_root={args.output_root} "
        f"steps={args.steps} checkpoint_every={args.checkpoint_every} "
        f"gpus={','.join(gpu_ids)}",
        flush=True,
    )

    processes = []
    for gpu_id, assigned_rules in zip(gpu_ids, assignments):
        process = Process(target=worker, args=(args, gpu_id, assigned_rules))
        process.start()
        processes.append(process)

    failed = False
    for process in processes:
        process.join()
        failed = failed or process.exitcode != 0
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
