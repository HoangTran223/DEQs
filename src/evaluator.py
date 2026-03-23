"""
Evaluation utilities for FRFD models.
"""

import torch
import os
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from peft import PeftModel
from rouge_score import rouge_scorer
from datasets import load_dataset
from typing import Dict, List, Tuple, Any
from tqdm.auto import tqdm
import json



def preprocess_test(examples: Dict[str, Any], tokenizer: AutoTokenizer, max_seq_length: int) -> Dict[str, Any]:
    """Preprocessing function for evaluation."""
    prompts = examples['prompt']
    responses = examples['output']
    for i in range(len(responses)):
        if type(responses[i]) is list:
            responses[i] = responses[i][0]

    tokenized_prompts = tokenizer(
        prompts, 
        max_length=max_seq_length, 
        padding="longest", 
        truncation=True
    )
    tokenized_prompts["prompt"] = prompts
    tokenized_prompts["response"] = responses
    return tokenized_prompts


class Evaluator: 
    def __init__(self, tokenizer_path: str, 
                 model_path: str | None = None,
                 sft_lora: str | None = None,
                 distilled_lora: str | None = None,
                 device: str = 'cuda', seeds: list[int] = [10,20,30,40,50]):
        self.device = device

        if model_path is not None:
            # self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
            self.model = AutoModelForCausalLM.from_pretrained(model_path)
            if sft_lora is not None:
                self.model = PeftModel.from_pretrained(
                    self.model,
                    sft_lora
                ).merge_and_unload()
            if distilled_lora is not None:
                self.model = PeftModel.from_pretrained(
                    self.model,
                    distilled_lora
                ).merge_and_unload()

            self.model.to(device)
        else:
            self.model = None

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.seeds = seeds
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    
    @torch.no_grad()
    def evaluate_benchmark_dataset(
        self, 
        dataset_path: str, 
        dataset_name: str,
        batch_size: int = 10,
        max_seq_length: int = 256,
        max_new_tokens: int = 384
    ) -> float:
        """
        Evaluate model on a single benchmark dataset
        
        Args:
            dataset_path: Path to the dataset file (JSONL format)
            dataset_name: Name of the dataset for logging
            batch_size: Batch size for evaluation
            max_seq_length: Maximum input sequence length  
            max_new_tokens: Maximum new tokens to generate
            
        Returns:
            ROUGE-L F1 score percentage
        """
        print(f"\nEvaluating on {dataset_name}...")
        
        # Load dataset
        if dataset_path.endswith('.jsonl'):
            dataset = load_dataset("json", data_files=dataset_path)['train']
        else:
            dataset = load_dataset(dataset_path, split="train")
        
        # Preprocess dataset using the existing preprocess_test function
        processed_dataset = dataset.map(
            lambda x: preprocess_test(x, self.tokenizer, max_seq_length),
            batched=True,
            batch_size=batch_size
        )
        
        processed_dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "prompt", "response"]
        )
        
        # Create dataloader
        dataloader = DataLoader(
            processed_dataset,
            batch_size=batch_size,
            shuffle=False
        )
        
        # Evaluate
        self.model.eval()
        total_rouge_l = 0.0
        
        for seed in self.seeds:
            set_seed(seed)
            per_seed_samples = 0
            per_seed_rouge_l = 0
            with torch.no_grad():
                for batch in tqdm(dataloader, desc=f"Evaluating {dataset_name}"):
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    prompts = batch['prompt']
                    reference_responses = batch['response']
                    
                    # Generate responses
                    generated_responses = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens,
                        pad_token_id=self.tokenizer.eos_token_id,
                        do_sample=True,
                        temperature=1.0, # default
                        top_p=1.0, # default
                    )
                    
                    # Decode responses
                    decoded_responses = self.tokenizer.batch_decode(
                        generated_responses,
                        skip_special_tokens=True
                    )
                    
                    # Calculate ROUGE-L scores
                    for i in range(len(decoded_responses)):
                        prompt = prompts[i]
                        generated_response = decoded_responses[i]
                        reference_response = reference_responses[i]
                        
                        # Remove prompt from generated response
                        if generated_response.startswith(prompt):
                            generated_response = generated_response[len(prompt):].strip()
                        else:
                            # Fallback: try to find where response starts
                            response_start = generated_response.find("Response:")
                            if response_start != -1:
                                generated_response = generated_response[response_start + len("Response:"):].strip()
                            else:
                                generated_response = None
                        
                        # Remove special tokens from reference
                        reference_response = reference_response.replace('<pad>', '').replace('<|endoftext|>', '').strip()
                        
                        # Calculate ROUGE-L score if both responses are non-empty
                        if generated_response and reference_response:
                            score = self.rouge_scorer.score(
                                generated_response, 
                                reference_response
                            )['rougeL'].fmeasure
                            per_seed_rouge_l += score
                        if reference_response:
                            per_seed_samples += 1
            
            if per_seed_samples > 0:
                per_seed_rouge_l = per_seed_rouge_l * 100 / per_seed_samples
            else:
                per_seed_rouge_l = 0
            total_rouge_l += per_seed_rouge_l

            print(f"{dataset_name} - Seed {seed} ROUGE-L F1: {total_rouge_l:.2f}%")
        
        # Calculate final score
        total_rouge_l /= len(self.seeds)

        self.model.train()
        
        print(f"{dataset_name} ROUGE-L F1: {total_rouge_l:.2f}%")
        return total_rouge_l
    
    @torch.no_grad()
    def evaluate_multiple_benchmarks(
        self,
        benchmark_configs: Dict[str, str],
        batch_size: int = 10,
        max_seq_length: int = 256,
        max_new_tokens: int = 384
    ) -> Dict[str, Dict]:
        """
        Evaluate model on multiple benchmark datasets
        
        Args:
            benchmark_configs: Dictionary mapping dataset keys to file paths
                Example: {
                    "dolly": "/path/to/dolly/valid.jsonl",
                    "self_instruct": "/path/to/self_instruct/valid.jsonl"
                }
            batch_size: Batch size for evaluation
            max_seq_length: Maximum input sequence length
            max_new_tokens: Maximum new tokens to generate
            
        Returns:
            Dictionary with results for each benchmark
        """
        results = {}
        
        # Dataset name mapping
        dataset_names = {
            "dolly": "Dolly",
            "self_instruct": "Self-Instruct", 
            "vicuna": "Vicuna",
            "sni": "S-NI",
            "unni": "UnNI"
        }
        
        for key, dataset_path in benchmark_configs.items():
            dataset_name = dataset_names.get(key, key.title())
            
            if dataset_path and os.path.exists(dataset_path):
                try:
                    score = self.evaluate_benchmark_dataset(
                        dataset_path=dataset_path,
                        dataset_name=dataset_name,
                        batch_size=batch_size,
                        max_seq_length=max_seq_length,
                        max_new_tokens=max_new_tokens
                    )
                    results[key] = {
                        "dataset_name": dataset_name,
                        "dataset_path": dataset_path,
                        "rouge_l_f1": score,
                        "status": "success"
                    }
                except Exception as e:
                    print(f"Error evaluating {dataset_name}: {str(e)}")
                    results[key] = {
                        "dataset_name": dataset_name,
                        "dataset_path": dataset_path,
                        "rouge_l_f1": None,
                        "status": "error",
                        "error_message": str(e)
                    }
            else:
                print(f"Warning: Dataset path for {dataset_name} not found: {dataset_path}")
                results[key] = {
                    "dataset_name": dataset_name,
                    "dataset_path": dataset_path,
                    "rouge_l_f1": None,
                    "status": "not_found"
                }
        
        return results

    @torch.no_grad()
    def generate_and_save_outputs(
        self,
        dataset_path: str,
        output_file: str,
        batch_size: int = 10,
        max_seq_length: int = 256,
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        top_p: float = 1.0
    ):
        print(f"\nGenerating outputs for {dataset_path}...")
        
        # Load dataset
        if dataset_path.endswith('.jsonl'):
            dataset = load_dataset("json", data_files=dataset_path)['train']
        else:
            dataset = load_dataset(dataset_path, split="train")


        # Preprocess
        processed_dataset = dataset.map(
            lambda x: preprocess_test(x, self.tokenizer, max_seq_length),
            batched=True,
            batch_size=batch_size
        )
    
        processed_dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "prompt"]
        )
    
        dataloader = DataLoader(processed_dataset, batch_size=batch_size, shuffle=False)
    
        self.model.eval()
        generations = []
        # set_seed(42)
        set_seed(30)
    
        for batch in tqdm(dataloader, desc="Generating"):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            prompts = batch["prompt"]
    
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
    
            decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            for p, gen in zip(prompts, decoded):
                # cắt prompt ra để chỉ giữ phần model sinh
                if gen.startswith(p):
                    gen = gen[len(p):].strip()
                generations.append({"prompt": p, "generated_text": gen})

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            for item in generations:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
        print(f"Saved {len(generations)} generations to {output_file}")