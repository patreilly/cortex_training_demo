"""Probe which label convention the server's SFT loss expects (forward only, no step)."""
import os
import json
import torch
from cortex_training import CortexTrainingClient, Hardware, SubJobConfig, serialize_forward_backward_args
import train as T

ex = sorted((json.loads(l) for l in open(T.HERE / "data" / "train.jsonl")), key=lambda e: e["n_tokens"])[0]
ids = ex["input_ids"]
shifted = ex["labels"]                                     # labels[i] = ids[i+1]
unshifted = [-100] + shifted[:-1]                          # labels[i] = ids[i] (HF style)
c = CortexTrainingClient.from_configs(os.environ.get("SNOWFLAKE_CONNECTION_NAME", "default"), database="CT_DEMO_DB", schema="CORTEX_TRAINING")
sj = SubJobConfig.training_job("Qwen/Qwen3.5-4B", optimizer={"name": "AdamW", "lr": 1e-4}, max_seq_len=8192,
                               train_batch_size=1, n_gpus=1, dtype="bfloat16",
                               extra_training={"peft_config": {"peft_type": "Lora", "r": 8, "lora_alpha": 16,
                                                               "bias": "none", "target_modules": T.LORA_TARGETS}})
job = c.create_job([sj], hardware=Hardware.H200, idle_timeout_seconds=600)
print("job", job, flush=True)
try:
    c.wait_for_job(job)
    for name, lab in [("pre_shifted", shifted), ("hf_unshifted", unshifted), ("pre_shifted_again", shifted)]:
        p = serialize_forward_backward_args(kwargs={"input_ids": torch.tensor([ids]), "labels": torch.tensor([lab])})
        r = c.poll_request(job, c.forward_backward(job, p))
        print(name, T.scalar_metrics(r), flush=True)
finally:
    c.cancel_job(job)
