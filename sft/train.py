"""LoRA SFT of a Qwen student on converted agent traces with Cortex Training.

Client-side loop:  create_job -> wait running -> [forward_backward x grad_accum
-> step] x N -> save (weights-only) -> cancel.  One example per forward_backward
call (batch of 1, no padding); labels are HF-style (unshifted, the server
shifts) with non-assistant tokens = -100 (built by convert.py).

  python sft/train.py --smoke                 # 2 steps, proves the pipeline
  python sft/train.py --epochs 2 --run-name student_qwen35_4b_v1
"""

import os
import argparse
import json
import random
import time
from pathlib import Path

import torch
from cortex_training import CortexTrainingClient, Hardware, SubJobConfig, serialize_forward_backward_args

HERE = Path(__file__).resolve().parent
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load_examples(path):
    return [json.loads(l) for l in open(path)]


def payload(ex):
    ids = torch.tensor([ex["input_ids"]], dtype=torch.long)
    labels = torch.tensor([ex["labels"]], dtype=torch.long)
    return serialize_forward_backward_args(kwargs={"input_ids": ids, "labels": labels})


def scalar_metrics(result):
    """Pull loss-like scalars out of whatever the server returns."""
    out = {}

    def walk(obj, prefix=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                walk(v, f"{prefix}{k}.")
        elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
            out[prefix[:-1]] = obj
        elif torch.is_tensor(obj) and obj.numel() == 1:
            out[prefix[:-1]] = obj.item()

    walk(result)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--connection", default=os.environ.get("SNOWFLAKE_CONNECTION_NAME", "default"))
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--data", default=str(HERE / "data" / "train.jsonl"))
    ap.add_argument("--run-name", default="student_qwen35_4b_smoke")
    ap.add_argument("--max-seq-len", type=int, default=32768)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--save-every", type=int, default=50, help="optimizer steps")
    ap.add_argument("--smoke", action="store_true", help="2 steps x 2 examples, then save")
    ap.add_argument("--max-steps", type=int, default=0, help="with --smoke: overfit the 4 shortest examples for N steps")
    ap.add_argument("--keep-job", action="store_true", help="don't cancel the job at the end")
    args = ap.parse_args()

    examples = load_examples(args.data)
    random.Random(0).shuffle(examples)
    if args.smoke:
        examples = sorted(examples, key=lambda e: e["n_tokens"])[:4]
        args.grad_accum, total_steps = 2, (args.max_steps or 2)
    else:
        total_steps = int(len(examples) * args.epochs) // args.grad_accum

    client = CortexTrainingClient.from_configs(args.connection, database="CT_DEMO_DB", schema="CORTEX_TRAINING")
    sub_job = SubJobConfig.training_job(
        args.model,
        optimizer={"name": "AdamW", "lr": args.lr, "weight_decay": 0.0},
        max_seq_len=args.max_seq_len,
        train_batch_size=1,
        n_gpus=1,
        gradient_clipping=1.0,
        dtype="bfloat16",
        seed=0,
        extra_training={"peft_config": {
            "peft_type": "Lora", "r": args.rank, "lora_alpha": 2 * args.rank,
            "lora_dropout": 0.0, "bias": "none", "target_modules": LORA_TARGETS}},
    )
    # experiment_name must name an existing experiment; omit it so the server auto-creates one.
    job_id = client.create_job([sub_job], hardware=Hardware.H200, idle_timeout_seconds=900)
    log_path = HERE / "runs" / f"{args.run_name}.jsonl"
    log_path.parent.mkdir(exist_ok=True)
    print(f"job {job_id}: {len(examples)} examples, {total_steps} steps, waiting for GPUs...", flush=True)
    t0 = time.time()
    checkpoints = []
    try:
        client.wait_for_job(job_id)
        print(f"running after {time.time() - t0:.0f}s", flush=True)
        log = open(log_path, "a")
        i = 0
        for step in range(1, total_steps + 1):
            ts, tok_trained, metrics = time.time(), 0, []
            for _ in range(args.grad_accum):
                ex = examples[i % len(examples)]
                i += 1
                rid = client.forward_backward(job_id, payload(ex))
                metrics.append(scalar_metrics(client.poll_request(job_id, rid)))
                tok_trained += ex["n_trained"]
            client.poll_request(job_id, client.step(job_id, learning_rate=args.lr))
            loss_keys = [k for k in metrics[0] if "loss" in k.lower()]
            loss = {k: sum(m.get(k, 0) for m in metrics) / len(metrics) for k in loss_keys}
            rec = {"step": step, "epoch": round(i / len(examples), 3), "loss": loss,
                   "trained_tokens": tok_trained, "sec": round(time.time() - ts, 1)}
            if step == 1:
                rec["first_result_keys"] = sorted(metrics[0])
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(json.dumps(rec), flush=True)
            if step % args.save_every == 0 or step == total_steps:
                ckpt = f"{args.run_name}-step{step}"
                client.poll_request(job_id, client.save(job_id, checkpoint_id=ckpt, checkpoint_type="weights-only"))
                checkpoints.append(ckpt)
                print(f"saved checkpoint {ckpt}", flush=True)
    finally:
        info = {"job_id": job_id, "run_name": args.run_name, "model": args.model,
                "checkpoints": checkpoints, "elapsed_s": round(time.time() - t0)}
        json.dump(info, open(HERE / "runs" / f"{args.run_name}.job.json", "w"), indent=1)
        if not args.keep_job:
            client.cancel_job(job_id)
            print(f"cancelled job {job_id}", flush=True)
    print(json.dumps(info))


if __name__ == "__main__":
    main()
