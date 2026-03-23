import argparse
from arguments import Arguments
from evaluator import Evaluator
from transformers import AutoModelForCausalLM, HfArgumentParser

import torch
import json
import numpy as np
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def main():
    hf_parser = HfArgumentParser(Arguments)
    args, remaining = hf_parser.parse_args_into_dataclasses(return_remaining_strings=True)

    extra_parser = argparse.ArgumentParser(add_help=False)
    extra_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    extra_parser.add_argument("--model_path", type=str, default=None)
    extra_parser.add_argument("--lora_path", type=str, default=None)
    extra_parser.add_argument("--tokenizer", type=str, default=None)
    extra_parser.add_argument("--bf16", action="store_true")

    extras = extra_parser.parse_args(remaining)

    set_seed(extras.seed)

    if extras.lora_path is not None:
        evaluator = Evaluator(
            tokenizer_path=extras.tokenizer,
            model_path=extras.model_path,
            distilled_lora=extras.lora_path,
            device=args.student_device,
            seeds=[10, 20, 30, 40, 50]
        )
    else:
        evaluator = Evaluator(
            tokenizer_path=extras.tokenizer,
            model_path=extras.model_path,
            device=args.student_device,
            seeds=[10, 20, 30, 40, 50]
        )
    
    evaluator.model.config.output_hidden_states=False
    evaluator.model.config.output_attentions=False

    benchmark_configs = {'sni': './data/sinst/11_/valid.jsonl',
                        'dolly': './data/dolly/valid.jsonl',
                        'self_instruct': './data/self-inst/valid.jsonl',
                        'vicuna': './data/vicuna/valid.jsonl'
                        }

    # benchmark_configs = {'train': './data/dolly/train.jsonl',
    #                      'dev': './data/dolly/dev.jsonl'}

    for key in benchmark_configs:
        evaluator.generate_and_save_outputs(
            dataset_path=benchmark_configs[key],
            output_file=args.output_dir + f"/generated_{key}.jsonl",
            batch_size=args.val_batch_size,
            max_seq_length=256, max_new_tokens=512,
            temperature=0.9, top_p=1.0
        )

if __name__ == "__main__":
    main()