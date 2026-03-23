"""
DEQ + DistiLLM  Knowledge Distillation training script.

Pure KD approach: the DEQ student is trained to match the teacher's
output distribution.  No intermediate-layer alignment is used — only
the final logits from the DEQ equilibrium state z* are compared.

Loss:
    L = (1 - kd_ratio) * L_ce  +  kd_ratio * L_kd

    L_ce : cross-entropy on ground-truth labels (using z* logits)
    L_kd : skewed reverse-KL (or other KL variant) on teacher logits
"""

import time
import os
import math
import random
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW
import torch.distributed as dist
import deepspeed
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoConfig, GenerationConfig

from arguments import get_args
from data_utils.lm_datasets import LMTrainDataset
from utils import (
    get_optimizer_params,
    print_args,
    initialize,
    print_rank,
    get_rank,
    save_rank,
    all_gather,
    get_tokenizer,
)
from distillm import forward_kl, reverse_kl, js_distance, tv_distance
from distillm import skewed_forward_kl, skewed_reverse_kl
from distillm import SampleGenerator, ReplayBuffer
from rouge_metric import compute_metrics

from deq_model import DEQStudent, build_deq_config

torch.set_num_threads(4)


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def get_teacher_model(args, device):
    config = AutoConfig.from_pretrained(args.teacher_model_path)
    config.is_model_parallel = False
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.teacher_model_path, config=config,
            device_map={"": device}, torch_dtype=torch.float16,
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            args.teacher_model_path, config=config,
            device_map={"": device}, torch_dtype=torch.float32,
        )
        model = model.half()

    if dist.get_rank() == 0:
        print(f" > teacher parameters: {sum(p.nelement() for p in model.parameters())}", flush=True)
    model.eval()
    return model


def _count_params(model):
    return sum(p.nelement() for p in {id(p): p for p in model.parameters()}.values())


def _print_size_comparison(args, deq_model):
    """Print original LLM size vs DEQ-compressed size."""
    if dist.get_rank() != 0:
        return
    deq_params = _count_params(deq_model)
    try:
        orig = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float32)
        orig_params = _count_params(orig)
        del orig
        torch.cuda.empty_cache()
    except Exception:
        orig_params = None

    print("\n" + "=" * 60, flush=True)
    print("  MODEL SIZE COMPARISON", flush=True)
    print("=" * 60, flush=True)
    if orig_params:
        ratio = deq_params / orig_params * 100
        saved = (1 - deq_params / orig_params) * 100
        print(f"  Original student LLM : {orig_params:>14,} params", flush=True)
        print(f"  DEQ student          : {deq_params:>14,} params  ({ratio:.1f}%)", flush=True)
        print(f"  Reduction            : {saved:.1f}% fewer parameters", flush=True)
    else:
        print(f"  DEQ student          : {deq_params:>14,} params", flush=True)
    print(f"  DEQ blocks           : {deq_model.n_deq_blocks}", flush=True)
    print("=" * 60 + "\n", flush=True)


def get_deq_student(args, device):
    ckpt_flag = os.path.join(args.model_path, "deq_config.json")
    if os.path.exists(ckpt_flag):
        print_rank(f"Loading DEQ checkpoint from {args.model_path}")
        model = DEQStudent.from_pretrained(args.model_path, device=device)
    else:
        config = AutoConfig.from_pretrained(args.model_path)
        deq_kwargs = build_deq_config(args)
        model = DEQStudent(config, args.model_type, deq_kwargs, n_deq_blocks=args.n_deq_blocks)
        if args.deq_init_pretrained:
            print_rank("Initialising DEQ blocks from pretrained weights …")
            model.init_from_pretrained(args.model_path)

    dtype = torch.float32 if args.fp32 else torch.float16
    if args.bf16:
        dtype = torch.bfloat16
    model = model.to(dtype=dtype, device=device)

    _print_size_comparison(args, model)
    return model


# ---------------------------------------------------------------------------
# Optimizer / scheduler
# ---------------------------------------------------------------------------

def get_optimizer(args, model):
    param_groups = get_optimizer_params(args, model)
    return AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)


def get_learning_rate_scheduler(args, optimizer):
    from transformers import get_constant_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup
    from torch.optim.lr_scheduler import CosineAnnealingLR

    if args.total_iters is None:
        args.total_iters = args.train_iters_per_epoch * args.epochs
    if args.lr_decay_style == "constant":
        return get_constant_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_iters)
    if args.lr_decay_style == "cosine":
        return CosineAnnealingLR(optimizer, T_max=args.total_iters, eta_min=args.lr_min)
    if args.lr_decay_style == "noam":
        return get_polynomial_decay_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_iters,
            num_training_steps=args.total_iters, power=0.5,
        )
    raise ValueError(f"Unknown lr_decay_style: {args.lr_decay_style}")


def setup_model_and_optimizer(args, ds_config, device):
    model = get_deq_student(args, device)
    optimizer = get_optimizer(args, model)
    lr_scheduler = get_learning_rate_scheduler(args, optimizer)
    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model, optimizer=optimizer, args=args,
        lr_scheduler=lr_scheduler, mpu=None, config_params=ds_config,
    )
    print_rank("Model mem\n", torch.cuda.memory_summary())
    return model, optimizer, lr_scheduler


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def prepare_dataset(args, tokenizer):
    data = {}
    rng = random.Random(args.seed)
    if args.do_train:
        data["train"] = LMTrainDataset(args, tokenizer, args.data_dir, "train", args.train_num, args.train_ratio, rng)
        print_rank("train num", len(data["train"]))
        data["dev"] = LMTrainDataset(args, tokenizer, args.data_dir, "valid", args.dev_num, args.dev_ratio, rng)
    elif args.do_eval:
        data["test"] = LMTrainDataset(args, tokenizer, args.data_dir, "valid", args.dev_num, args.dev_ratio, rng)
    return data


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def get_distil_loss(args, teacher_logits, no_model_batch, logits):
    t = args.type
    if "sfkl" in t:
        return skewed_forward_kl(logits, teacher_logits, no_model_batch, lam=args.skew_alpha)
    if "srkl" in t:
        return skewed_reverse_kl(logits, teacher_logits, no_model_batch, lam=args.skew_alpha)
    if "jsd" in t:
        return js_distance(logits, teacher_logits, no_model_batch)
    if "tvd" in t:
        return tv_distance(logits, teacher_logits, no_model_batch)
    if "fkl" in t or t == "kd":
        return forward_kl(logits, teacher_logits, no_model_batch)
    if "rkl" in t:
        return reverse_kl(logits, teacher_logits, no_model_batch)
    raise NotImplementedError(f"Unknown distillation type: {t}")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _log_vram(tag, device=0):
    """Log current VRAM usage."""
    if dist.get_rank() != 0:
        return
    alloc = torch.cuda.memory_allocated(device) / 1024**3
    reserved = torch.cuda.memory_reserved(device) / 1024**3
    total = torch.cuda.get_device_properties(device).total_mem / 1024**3
    print(f"  [VRAM {tag}] allocated={alloc:.2f}GB  reserved={reserved:.2f}GB  total={total:.2f}GB", flush=True)


def finetune(args, tokenizer, model, optimizer, lr_scheduler, dataset, device, teacher_model=None):
    print_rank("Start DEQ Fine-tuning")

    dp_world_size = dist.get_world_size()
    dp_rank = dist.get_rank()
    loss_func = nn.CrossEntropyLoss(ignore_index=-100)

    sampler = DistributedSampler(dataset["train"], shuffle=True, drop_last=True, rank=dp_rank, num_replicas=dp_world_size)
    train_dataloader = DataLoader(
        dataset["train"], sampler=sampler, batch_size=args.batch_size,
        num_workers=args.num_workers, collate_fn=dataset["train"].collate,
    )

    student_generator = SampleGenerator(args, tokenizer)
    replay_buffer = ReplayBuffer(args)

    step, global_step = 1, 1
    total_loss, total_distil_loss, total_time = 0.0, 0.0, 0.0
    adaptive_threshold = args.init_threshold if "adaptive" in args.type else -1.0
    prev_avg_loss = evaluate(args, tokenizer, model, dataset["dev"], "dev", 0, device, adaptive_threshold)

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        model.train()

        for it, (model_batch, no_model_batch, gen_data) in enumerate(train_dataloader):
            dataset["train"].move_to_device(model_batch, no_model_batch, gen_data, device)
            torch.cuda.synchronize()
            st_time = time.time()

            # ---- adaptive sampling (same as DistiLLM baseline) ----
            if "adaptive" in args.type and adaptive_threshold > 0:
                if args.replay_ratio == "constant":
                    samp_threshold = adaptive_threshold * 0.5
                elif args.replay_ratio == "increasing":
                    samp_threshold = adaptive_threshold * global_step / args.total_iters
                else:
                    samp_threshold = adaptive_threshold * (1 - global_step / args.total_iters)
            else:
                samp_threshold = -1.0

            if args.student_gen:
                r = np.random.uniform(0, 1)
                if "mixed" in args.type and r < args.mixed_alpha:
                    model_batch = student_generator.run_sample(model, gen_data)
                    no_model_batch["label"] = model_batch.pop("no_model_batch")
                    replay_buffer.move_to_memory(model_batch, no_model_batch, gen_data)
                    model_batch, no_model_batch, gen_data = replay_buffer.sample()
                    model_batch, no_model_batch, gen_data = replay_buffer.move_to_device(model_batch, no_model_batch, gen_data, device)
                elif "adaptive" in args.type and (r < samp_threshold or (r < adaptive_threshold and len(replay_buffer) < args.capacity)):
                    model_batch = student_generator.run_sample(model, gen_data)
                    no_model_batch["label"] = model_batch.pop("no_model_batch")
                    if args.model_type in ["opt"]:
                        model_batch.pop("position_ids", None)
                    replay_buffer.move_to_memory(model_batch, no_model_batch, gen_data)
                elif "adaptive" in args.type and r < adaptive_threshold:
                    model_batch, no_model_batch, gen_data = replay_buffer.sample()
                    model_batch, no_model_batch, gen_data = replay_buffer.move_to_device(model_batch, no_model_batch, gen_data, device)
                model.train()

            # ---- student forward ----
            outputs = model(**model_batch, use_cache=False)
            logits = outputs.logits

            lm_loss = loss_func(logits.float().view(-1, logits.size(-1)),
                                no_model_batch["label"].view(-1))

            # ---- distillation ----
            if teacher_model is not None:
                with torch.no_grad():
                    teacher_model.eval()
                    teacher_logits = teacher_model(**model_batch, use_cache=False).logits

                distil_loss = get_distil_loss(args, teacher_logits, no_model_batch, logits)
                loss = (1 - args.kd_ratio) * lm_loss + args.kd_ratio * distil_loss
            else:
                distil_loss = torch.tensor(0.0)
                loss = lm_loss

            model.backward(loss)
            model.step()

            if step == 1:
                _log_vram("after first train step (peak)", device)

            # ---- logging ----
            dist.all_reduce(loss, dist.ReduceOp.SUM)
            global_loss = loss.item() / dp_world_size
            global_distil_loss = 0.0
            if teacher_model is not None:
                dist.all_reduce(distil_loss, dist.ReduceOp.SUM)
                global_distil_loss = distil_loss.item() / dp_world_size
                total_distil_loss += global_distil_loss

            torch.cuda.synchronize()
            elapsed_time = time.time() - st_time
            total_loss += global_loss
            total_time += elapsed_time

            def _log(ll, dl, tt):
                return (
                    f"train | epoch {epoch:3d} | Iter: {step:6d}/{args.total_iters * args.gradient_accumulation_steps:6d} "
                    f"| global iter: {global_step:6d}/{args.total_iters:6d} | loss: {ll:.4f} "
                    f"| ds_loss: {dl:.4f} | lr: {lr_scheduler.get_last_lr()[0]:.4e} "
                    f"| scale: {getattr(optimizer, 'cur_scale', 0):10.4f} "
                    f"| micro time: {elapsed_time:.3f} | step time: {tt:.3f}"
                )

            if args.mid_log_num > 0:
                ms = max(1, args.gradient_accumulation_steps // args.mid_log_num)
                if step % ms == 0:
                    print_rank(_log(global_loss, global_distil_loss, 0))

            if global_step % args.log_interval == 0 and step % args.gradient_accumulation_steps == 0:
                denom_l = args.log_interval * args.gradient_accumulation_steps
                log_str = _log(total_loss / denom_l, total_distil_loss / denom_l, total_time / args.log_interval)
                print_rank("*" * 100); print_rank(log_str); print_rank(args.save); print_rank("*" * 100)
                save_rank(log_str, os.path.join(args.save, "log.txt"))
                total_loss, total_distil_loss, total_time = 0.0, 0.0, 0.0

            # ---- checkpoint ----
            if args.save and args.save_interval and global_step % args.save_interval == 0 and step % args.gradient_accumulation_steps == 0:
                sd = os.path.join(args.save, str(global_step))
                if dist.get_rank() == 0:
                    os.makedirs(sd, exist_ok=True)
                    print_rank(f"Model save to {sd}")
                    tokenizer.save_pretrained(sd)
                    model.module.save_pretrained(sd)
                dist.barrier()

            # ---- eval ----
            if args.eval_interval and global_step % args.eval_interval == 0 and step % args.gradient_accumulation_steps == 0:
                curr = evaluate(args, tokenizer, model, dataset["dev"], "dev", epoch, device, adaptive_threshold)
                if "adaptive" in args.type and curr >= prev_avg_loss + args.loss_eps:
                    adaptive_threshold = min(adaptive_threshold + 0.1, 1.0)
                    prev_avg_loss = curr
                model.train()

            step += 1
            if step % args.gradient_accumulation_steps == 0:
                global_step += 1
            if global_step > args.total_iters:
                break

    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(args, tokenizer, model, dataset, split, epoch, device, adaptive_threshold=None):
    dp_world_size = dist.get_world_size()
    dp_rank = dist.get_rank()
    loss_func = nn.CrossEntropyLoss()

    gen_cfg = GenerationConfig(
        do_sample=args.do_sample, top_p=args.top_p, top_k=args.top_k,
        temperature=args.temperature, repetition_penalty=args.repetition_penalty,
        max_length=args.max_length, eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id, return_dict_in_generate=True, output_scores=False,
    )

    sampler = DistributedSampler(dataset, shuffle=False, drop_last=False, rank=dp_rank, num_replicas=dp_world_size)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.eval_batch_size,
                            num_workers=args.num_workers, collate_fn=dataset.collate)

    model.eval()
    all_loss, n_steps = 0.0, 0
    all_resp = []

    with torch.no_grad():
        for it, (mb, nmb, gd) in enumerate(tqdm(dataloader, desc="Evaluating", disable=(dist.get_rank() != 0))):
            dataset.move_to_device(mb, nmb, gd, device)
            logits = model(**mb).logits
            loss = loss_func(logits.view(-1, logits.size(-1)), nmb["label"].view(-1))

            if args.eval_gen:
                mnt = args.max_length - gd["input_ids"].size(1)
                gen_out = model.module.generate(
                    input_ids=gd["input_ids"], attention_mask=gd["attention_mask"],
                    generation_config=gen_cfg, max_new_tokens=mnt,
                )
                full = gen_out.sequences
                full = F.pad(full, (0, args.max_length - full.shape[1]), value=tokenizer.pad_token_id)
                all_resp.append(full[:, gd["input_ids"].size(1):])

            dist.all_reduce(loss, dist.ReduceOp.SUM)
            all_loss += loss.item() / dp_world_size
            n_steps += 1

    if args.eval_gen:
        all_resp = torch.cat(all_resp, 0)
        all_resp = all_gather(all_resp, dim=1, world_size=dp_world_size, op="stack").view(-1, all_resp.size(-1))
        responses = tokenizer.batch_decode(all_resp, skip_special_tokens=True)

    if get_rank() == 0:
        res = {}
        if args.eval_gen:
            refs = dataset.answers
            responses = responses[:len(refs)]
            res = compute_metrics(responses, refs)
            ed = os.path.join(args.save, "eval", str(epoch))
            os.makedirs(ed, exist_ok=True)
            with open(os.path.join(ed, "answers.jsonl"), "w") as f:
                for r in responses:
                    f.write(json.dumps({"text": r}) + "\n")
        avg = all_loss / n_steps
        ls = f"{split} | avg_loss: {avg} | {res}"
        if adaptive_threshold is not None and adaptive_threshold > 0:
            ls += f" | threshold: {adaptive_threshold}"
        print_rank(ls)
        save_rank(ls, os.path.join(args.save, "log.txt"))

    return all_loss / n_steps


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    torch.backends.cudnn.enabled = False
    args = get_args()
    initialize(args)

    if dist.get_rank() == 0:
        print_args(args)
        with open(os.path.join(args.save, "args.json"), "w") as f:
            json.dump(vars(args), f, default=str)

    device = torch.cuda.current_device()
    save_rank(
        "\n\n" + "=" * 30 + f" DEQ EXP at {time.strftime('%Y-%m-%d %H:%M:%S')} " + "=" * 30,
        os.path.join(args.save, "log.txt"),
    )

    with open(args.deepspeed_config, "r") as f:
        ds_config = json.load(f)
    ds_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    ds_config["train_micro_batch_size_per_gpu"] = args.batch_size
    ds_config["gradient_clipping"] = args.clip_grad
    ds_config["steps_per_print"] = 10_000_000
    if not args.do_train:
        ds_config["zero_optimization"]["stage"] = 0
    args.fp32 = not ds_config["fp16"]["enabled"]
    args.bf16 = ds_config.get("bf16", {}).get("enabled", False)
    args.deepspeed_config = None

    tokenizer = get_tokenizer(args)
    dataset = prepare_dataset(args, tokenizer)
    dp_world_size = dist.get_world_size()

    if args.do_train:
        args.train_iters_per_epoch = int(len(dataset["train"]) / (args.batch_size * dp_world_size * args.gradient_accumulation_steps))
        print_rank("Train iters per epoch", args.train_iters_per_epoch)
        if args.total_iters is None:
            args.total_iters = args.train_iters_per_epoch * args.epochs
        if args.epochs is None:
            args.epochs = math.ceil(args.total_iters / args.train_iters_per_epoch)
        print_rank("total_iters", args.total_iters)
        if args.save_interval == -1:
            args.save_interval = args.train_iters_per_epoch
        if args.eval_interval == -1:
            args.eval_interval = args.train_iters_per_epoch

    model, optimizer, lr_scheduler = setup_model_and_optimizer(args, ds_config, device)
    _log_vram("after DEQ student loaded", device)

    if args.teacher_model_type is None:
        args.teacher_model_type = args.model_type
    teacher_model = get_teacher_model(args, device) if args.teacher_model_path else None
    _log_vram("after teacher loaded", device)

    if args.do_train:
        finetune(args, tokenizer, model, optimizer, lr_scheduler, dataset, device, teacher_model=teacher_model)
    if args.do_eval:
        evaluate(args, tokenizer, model, dataset["test"], "test", 0, device)


if __name__ == "__main__":
    main()
