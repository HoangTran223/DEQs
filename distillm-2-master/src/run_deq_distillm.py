"""
DistiLLM-2 + DEQ Student training script.

Uses the DistiLLMTrainer (HuggingFace Trainer-based) with a DEQ student model.
The DEQ model only provides final logits — no intermediate hidden states are
needed by DistiLLM-2's loss computation (distillm_v2, skewed KL).
"""

import logging
import sys
import os

import torch
import transformers
from transformers import AutoModelForCausalLM, set_seed
from datasets import load_dataset

# Add distillm-fdd to path so we can import the DEQ model
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "distillm-fdd"))

from alignment import (
    DataArguments,
    DPOConfig,
    H4ArgumentParser,
    ModelArguments,
    get_checkpoint,
    get_datasets,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    get_tokenizer,
    is_adapter_model,
)
from distillm_trainer import DistiLLMTrainer
from deq_model import DEQStudent, build_deq_config

logger = logging.getLogger(__name__)


def add_deq_fields(parser):
    """Add DEQ-specific CLI arguments on top of the standard H4 parser."""
    parser.add_argument("--n_deq_blocks", type=int, default=2)
    parser.add_argument("--deq_f_max_iter", type=int, default=30)
    parser.add_argument("--deq_b_max_iter", type=int, default=30)
    parser.add_argument("--deq_f_tol", type=float, default=1e-3)
    parser.add_argument("--deq_b_tol", type=float, default=1e-6)
    parser.add_argument("--deq_n_states", type=int, default=1)
    parser.add_argument("--deq_gamma", type=float, default=0.8)
    parser.add_argument("--deq_solver", type=str, default="fixed_point_iter")
    parser.add_argument("--deq_norm_type", type=str, default="weight_norm")
    parser.add_argument("--deq_init_pretrained", action="store_true")
    parser.add_argument("--deq_model_type", type=str, default="gpt2",
                        choices=["gpt2", "qwen", "opt"])


def _build_deq_kwargs(extra):
    return {
        "f_solver": extra.deq_solver,
        "b_solver": extra.deq_solver,
        "f_max_iter": extra.deq_f_max_iter,
        "b_max_iter": extra.deq_b_max_iter,
        "f_tol": extra.deq_f_tol,
        "b_tol": extra.deq_b_tol,
        "f_stop_mode": "abs",
        "b_stop_mode": "abs",
        "core": "sliced",
        "n_states": extra.deq_n_states,
        "grad": [1],
        "gamma": extra.deq_gamma,
        "norm_type": extra.deq_norm_type,
        "ift": False,
        "hook_ift": False,
        "tau": 1.0,
    }


def main():
    parser = H4ArgumentParser((ModelArguments, DataArguments, DPOConfig))
    add_deq_fields(parser)
    model_args, data_args, training_args, extra = parser.parse()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")
    logger.info(f"DEQ parameters: blocks={extra.n_deq_blocks}, iter={extra.deq_f_max_iter}, "
                f"model_type={extra.deq_model_type}")

    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    set_seed(training_args.seed)

    # ---- datasets ----
    raw_datasets = get_datasets(
        data_args,
        splits=data_args.dataset_splits,
        configs=data_args.dataset_configs,
        columns_to_keep=["chosen", "rejected", "prompt"],
    )
    logger.info(
        f"Training on: {[s + ' : ' + str(d.num_rows) for s, d in raw_datasets.items()]}"
    )

    tokenizer = get_tokenizer(model_args, data_args)

    # ---- build DEQ student ----
    from transformers import AutoConfig

    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )

    student_config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    deq_kwargs = _build_deq_kwargs(extra)
    model = DEQStudent(student_config, extra.deq_model_type, deq_kwargs, extra.n_deq_blocks)

    if extra.deq_init_pretrained:
        logger.info("Initialising DEQ from pretrained weights …")
        model.init_from_pretrained(model_args.model_name_or_path)

    if torch_dtype not in ["auto", None]:
        model = model.to(dtype=torch_dtype)

    deq_params = sum(p.nelement() for p in {id(p): p for p in model.parameters()}.values())
    try:
        orig = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, torch_dtype=torch.float32)
        orig_params = sum(p.nelement() for p in {id(p): p for p in orig.parameters()}.values())
        del orig
        ratio = deq_params / orig_params * 100
        saved = 100 - ratio
        logger.info("=" * 60)
        logger.info("  MODEL SIZE COMPARISON")
        logger.info("=" * 60)
        logger.info(f"  Original student LLM : {orig_params:>14,} params")
        logger.info(f"  DEQ student          : {deq_params:>14,} params  ({ratio:.1f}%)")
        logger.info(f"  Reduction            : {saved:.1f}% fewer parameters")
        logger.info(f"  DEQ blocks           : {extra.n_deq_blocks}")
        logger.info("=" * 60)
    except Exception:
        logger.info(f"DEQ student unique params: {deq_params:,}")

    # ---- teacher (ref_model) ----
    ref_model = model_args.ref_model_name_or_path
    quantization_config = get_quantization_config(model_args)
    ref_model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    # ---- trainer ----
    trainer = DistiLLMTrainer(
        model,
        ref_model,
        model_init_kwargs=None,
        ref_model_init_kwargs=ref_model_kwargs,
        args=training_args,
        beta=training_args.beta,
        train_dataset=raw_datasets["train"],
        eval_dataset=raw_datasets.get("test"),
        tokenizer=tokenizer,
        max_length=training_args.max_length,
        max_prompt_length=training_args.max_prompt_length,
        peft_config=get_peft_config(model_args),
        loss_type=training_args.loss_type,
        force_use_ref_model=True,
    )

    # ---- train ----
    checkpoint = training_args.resume_from_checkpoint or last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(raw_datasets["train"])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Training complete ***")

    # ---- save ----
    logger.info("*** Save model ***")
    save_dir = training_args.output_dir
    if trainer.accelerator.is_main_process:
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        trainer.model.config.use_cache = True
    logger.info(f"Model saved to {save_dir}")

    # ---- eval ----
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(raw_datasets.get("test", []))
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    logger.info("*** Done ***")


if __name__ == "__main__":
    os.environ["WANDB_DISABLED"] = "true"
    main()
