"""
DEQ Student Model for Knowledge Distillation.

Supports GPT-2, Qwen (Qwen2) and OPT model families.
Replaces the stack of N Decoder Layers with 1-2 weight-tied layers
solved via fixed-point iteration (Deep Equilibrium Model).

Architecture (all families):
    Embedding -> [DEQ: f_theta(blocks) iterates to z*] -> FinalNorm -> LM Head
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace

from transformers import AutoModelForCausalLM, AutoConfig
from torchdeq import get_deq
from torchdeq.norm import apply_norm, reset_norm, remove_norm


SUPPORTED_FAMILIES = ("gpt2", "qwen", "opt")


def build_deq_config(args):
    return {
        "f_solver": getattr(args, "deq_solver", "fixed_point_iter"),
        "b_solver": getattr(args, "deq_solver", "fixed_point_iter"),
        "f_max_iter": getattr(args, "deq_f_max_iter", 30),
        "b_max_iter": getattr(args, "deq_b_max_iter", 30),
        "f_tol": getattr(args, "deq_f_tol", 1e-3),
        "b_tol": getattr(args, "deq_b_tol", 1e-6),
        "f_stop_mode": "abs",
        "b_stop_mode": "abs",
        "core": "sliced",
        "n_states": getattr(args, "deq_n_states", 1),
        "grad": [1],
        "gamma": getattr(args, "deq_gamma", 0.8),
        "norm_type": getattr(args, "deq_norm_type", "weight_norm"),
        "ift": False,
        "hook_ift": False,
        "tau": 1.0,
    }


def _get_hidden_size(config, model_type):
    if model_type == "gpt2":
        return config.n_embd
    return config.hidden_size


def _import_block_class(model_type):
    if model_type == "gpt2":
        from transformers.models.gpt2.modeling_gpt2 import GPT2Block
        return GPT2Block
    elif model_type == "qwen":
        from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
        return Qwen2DecoderLayer
    elif model_type == "opt":
        from transformers.models.opt.modeling_opt import OPTDecoderLayer
        return OPTDecoderLayer
    raise ValueError(f"Unsupported model_type: {model_type}")


class DEQStudent(nn.Module):

    def __init__(self, config, model_type, deq_kwargs, n_deq_blocks=2):
        super().__init__()
        assert model_type in SUPPORTED_FAMILIES, f"{model_type} not in {SUPPORTED_FAMILIES}"

        if not hasattr(config, "_attn_implementation") or config._attn_implementation is None:
            config._attn_implementation = "eager"

        self.config = config
        self.model_type = model_type
        self.deq_kwargs = deq_kwargs if isinstance(deq_kwargs, dict) else dict(deq_kwargs)
        self.n_deq_blocks = n_deq_blocks

        hidden = _get_hidden_size(config, model_type)

        self._build_embedding(config, model_type, hidden)

        BlockCls = _import_block_class(model_type)
        self.deq_blocks = nn.ModuleList([BlockCls(config, layer_idx=i) for i in range(n_deq_blocks)])

        self.input_inject = nn.Linear(hidden, hidden)
        nn.init.normal_(self.input_inject.weight, std=0.02)
        nn.init.zeros_(self.input_inject.bias)

        apply_norm(self.deq_blocks, args=deq_kwargs)
        apply_norm(self.input_inject, args=deq_kwargs)
        self.deq = get_deq(deq_kwargs)

        self._build_output(config, model_type, hidden)

    # ------------------------------------------------------------------ #
    #  Construction helpers                                                #
    # ------------------------------------------------------------------ #

    def _build_embedding(self, config, mt, hidden):
        if mt == "gpt2":
            self.embed_tokens = nn.Embedding(config.vocab_size, config.n_embd)
            self.embed_positions = nn.Embedding(config.n_positions, config.n_embd)
            self.embed_drop = nn.Dropout(config.embd_pdrop)
        elif mt == "qwen":
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
            from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
            self.rotary_emb = Qwen2RotaryEmbedding(config=config)
        elif mt == "opt":
            from transformers.models.opt.modeling_opt import OPTLearnedPositionalEmbedding
            pad_id = getattr(config, "pad_token_id", 1)
            self.embed_tokens = nn.Embedding(config.vocab_size, config.word_embed_proj_dim, pad_id)
            self.embed_positions = OPTLearnedPositionalEmbedding(config.max_position_embeddings, config.hidden_size)
            self.project_in = (
                nn.Linear(config.word_embed_proj_dim, config.hidden_size, bias=False)
                if config.word_embed_proj_dim != config.hidden_size else None
            )

    def _build_output(self, config, mt, hidden):
        if mt == "gpt2":
            self.final_norm = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
            self.lm_head.weight = self.embed_tokens.weight
        elif mt == "qwen":
            from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
            self.final_norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            self.lm_head.weight = self.embed_tokens.weight
        elif mt == "opt":
            self.project_out = (
                nn.Linear(config.hidden_size, config.word_embed_proj_dim, bias=False)
                if config.word_embed_proj_dim != config.hidden_size else None
            )
            do_ln = getattr(config, "do_layer_norm_before", True)
            self.final_norm = nn.LayerNorm(config.hidden_size) if do_ln else None
            out_dim = config.word_embed_proj_dim
            self.lm_head = nn.Linear(out_dim, config.vocab_size, bias=False)

    # ------------------------------------------------------------------ #
    #  Forward internals                                                   #
    # ------------------------------------------------------------------ #

    def _compute_position_ids(self, input_ids, attention_mask):
        bsz, seq_len = input_ids.shape
        device = input_ids.device
        if attention_mask is not None:
            pos = attention_mask.long().cumsum(-1) - 1
            pos.masked_fill_(attention_mask == 0, 0)
            return pos
        return torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)

    def _embed(self, input_ids, attention_mask, position_ids):
        mt = self.model_type
        if mt == "gpt2":
            if position_ids is None:
                position_ids = self._compute_position_ids(input_ids, attention_mask)
            x = self.embed_tokens(input_ids) + self.embed_positions(position_ids)
            x = self.embed_drop(x)
            return x, {}
        if mt == "qwen":
            x = self.embed_tokens(input_ids)
            if position_ids is None:
                position_ids = self._compute_position_ids(input_ids, attention_mask)
            rotary = self.rotary_emb(x, position_ids)
            return x, {"position_ids": position_ids, "position_embeddings": rotary}
        # opt
        x = self.embed_tokens(input_ids)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        pos = self.embed_positions(attention_mask)
        x = x + pos
        if self.project_in is not None:
            x = self.project_in(x)
        return x, {}

    def _prepare_mask(self, attention_mask, seq_len, dtype, device):
        if attention_mask is None:
            return None
        if self.model_type == "gpt2":
            m = attention_mask[:, None, None, :].to(dtype=dtype)
            return (1.0 - m) * torch.finfo(dtype).min
        bsz = attention_mask.size(0)
        min_val = torch.finfo(dtype).min
        causal = torch.zeros(seq_len, seq_len, device=device, dtype=dtype)
        causal.masked_fill_(
            torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1),
            min_val,
        )
        causal = causal[None, None, :, :].expand(bsz, 1, -1, -1)
        pad = (1.0 - attention_mask[:, None, None, :].to(dtype)) * min_val
        return causal + pad

    @staticmethod
    def _extract_hidden(block_output):
        if isinstance(block_output, (tuple, list)):
            return block_output[0]
        return block_output

    @staticmethod
    def _linear_stored_dtype(linear_mod):
        """Dtype of underlying weights (torchdeq weight_norm / spectral_norm use weight_v or weight_orig)."""
        if hasattr(linear_mod, "weight_v"):
            return linear_mod.weight_v.dtype
        if hasattr(linear_mod, "weight_orig"):
            return linear_mod.weight_orig.dtype
        w = getattr(linear_mod, "weight", None)
        if isinstance(w, torch.Tensor):
            return w.dtype
        return next(linear_mod.parameters()).dtype

    def _run_blocks(self, h, mask, bk):
        mt = self.model_type
        for block in self.deq_blocks:
            if mt == "qwen":
                out = block(h, attention_mask=mask,
                            position_ids=bk.get("position_ids"),
                            position_embeddings=bk.get("position_embeddings"))
            else:
                out = block(h, attention_mask=mask)
            h = self._extract_hidden(out)
        return h

    def _to_logits(self, z_star):
        h = z_star
        if self.final_norm is not None:
            h = self.final_norm(h)
        if self.model_type == "opt" and getattr(self, "project_out", None) is not None:
            h = self.project_out(h)
        return self.lm_head(h)

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def forward(self, input_ids, attention_mask=None, position_ids=None, **kwargs):
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        x, block_kwargs = self._embed(input_ids, attention_mask, position_ids)

        if attention_mask is None:
            attention_mask = torch.ones(bsz, seq_len, device=device)

        # DEQ solver runs in fp32 to avoid fp16 accumulation errors across iterations
        compute_dtype = x.dtype
        x_f32 = x.float()
        mask_f32 = self._prepare_mask(attention_mask, seq_len, torch.float32, device)

        bk_f32 = {}
        for k, v in block_kwargs.items():
            if isinstance(v, torch.Tensor):
                bk_f32[k] = v
            elif isinstance(v, (tuple, list)):
                bk_f32[k] = tuple(t.float() if t.is_floating_point() else t for t in v)
            else:
                bk_f32[k] = v

        z_init = torch.zeros_like(x_f32)
        reset_norm(self.deq_blocks)
        reset_norm(self.input_inject)

        # Solver uses fp32 activations (x_f32, z). Match block/inject dtypes; embeddings may be fp32 under DeepSpeed.
        deq_w_dtype = self._linear_stored_dtype(self.input_inject)
        cast_deq_to_f32 = deq_w_dtype in (torch.float16, torch.bfloat16)
        if cast_deq_to_f32:
            self.deq_blocks.float()
            self.input_inject.float()
            # weight_norm / spectral_norm cache `weight` as a plain tensor; .float() only moves Parameters.
            reset_norm(self.deq_blocks)
            reset_norm(self.input_inject)

        def deq_func(z):
            return self._run_blocks(z + self.input_inject(x_f32), mask_f32, bk_f32)

        z_out, info = self.deq(deq_func, z_init)

        if cast_deq_to_f32:
            self.deq_blocks.to(deq_w_dtype)
            self.input_inject.to(deq_w_dtype)
            reset_norm(self.deq_blocks)
            reset_norm(self.input_inject)

        z_star = z_out[-1].to(compute_dtype)
        logits = self._to_logits(z_star)

        return SimpleNamespace(logits=logits, z_trajectory=z_out, info=info)

    # ---- generation -------------------------------------------------- #

    @torch.no_grad()
    def generate(self, input_ids, attention_mask=None, generation_config=None,
                 max_new_tokens=None, max_length=None, **kwargs):

        if generation_config is not None:
            temp = getattr(generation_config, "temperature", None) or 1.0
            top_k = getattr(generation_config, "top_k", None) or 0
            top_p = getattr(generation_config, "top_p", None) or 1.0
            do_sample = getattr(generation_config, "do_sample", False)
            eos_id = getattr(generation_config, "eos_token_id", None)
            pad_id = getattr(generation_config, "pad_token_id", None)
            cfg_ml = getattr(generation_config, "max_length", None)
        else:
            temp = kwargs.get("temperature", 1.0)
            top_k = kwargs.get("top_k", 0)
            top_p = kwargs.get("top_p", 1.0)
            do_sample = kwargs.get("do_sample", False)
            eos_id = kwargs.get("eos_token_id", None)
            pad_id = kwargs.get("pad_token_id", None)
            cfg_ml = kwargs.get("max_length", None)

        if max_new_tokens is None:
            ml = max_length or cfg_ml or (input_ids.size(1) + 128)
            max_new_tokens = ml - input_ids.size(1)
        max_new_tokens = max(max_new_tokens, 1)

        self.eval()
        ids = input_ids.clone()
        msk = attention_mask.clone() if attention_mask is not None else torch.ones_like(input_ids)
        done = torch.zeros(ids.size(0), dtype=torch.bool, device=ids.device)

        for _ in range(max_new_tokens):
            out = self.forward(ids, attention_mask=msk)
            nxt = out.logits[:, -1, :].float() / max(temp, 1e-8)

            if top_k > 0:
                v = torch.topk(nxt, top_k)[0]
                nxt[nxt < v[:, -1:]] = float("-inf")
            if 0 < top_p < 1.0:
                s_logits, s_idx = torch.sort(nxt, descending=True)
                cum = torch.cumsum(F.softmax(s_logits, dim=-1), dim=-1)
                rem = cum > top_p
                rem[:, 1:] = rem[:, :-1].clone()
                rem[:, 0] = False
                for b in range(nxt.size(0)):
                    nxt[b, s_idx[b][rem[b]]] = float("-inf")

            if do_sample:
                probs = F.softmax(nxt, dim=-1)
                tok = torch.multinomial(probs, 1)
            else:
                tok = nxt.argmax(-1, keepdim=True)

            if pad_id is not None:
                tok = tok.masked_fill(done.unsqueeze(1), pad_id)
            ids = torch.cat([ids, tok], dim=-1)
            msk = torch.cat([msk, (~done).long().unsqueeze(1)], dim=-1)
            if eos_id is not None:
                done = done | (tok.squeeze(-1) == eos_id)
                if done.all():
                    break

        return SimpleNamespace(sequences=ids)

    # ---- init from pretrained ---------------------------------------- #

    def init_from_pretrained(self, model_path):
        remove_norm(self.deq_blocks)
        remove_norm(self.input_inject)

        pretrained = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)
        mt = self.model_type

        if mt == "gpt2":
            self.embed_tokens.load_state_dict(pretrained.transformer.wte.state_dict())
            self.embed_positions.load_state_dict(pretrained.transformer.wpe.state_dict())
            self.final_norm.load_state_dict(pretrained.transformer.ln_f.state_dict())
            src_blocks = pretrained.transformer.h
        elif mt == "qwen":
            self.embed_tokens.load_state_dict(pretrained.model.embed_tokens.state_dict())
            self.final_norm.load_state_dict(pretrained.model.norm.state_dict())
            if hasattr(pretrained.model, "rotary_emb"):
                self.rotary_emb.load_state_dict(pretrained.model.rotary_emb.state_dict())
            src_blocks = pretrained.model.layers
        else:
            self.embed_tokens.load_state_dict(pretrained.model.decoder.embed_tokens.state_dict())
            self.embed_positions.load_state_dict(pretrained.model.decoder.embed_positions.state_dict())
            if pretrained.model.decoder.project_in is not None and self.project_in is not None:
                self.project_in.load_state_dict(pretrained.model.decoder.project_in.state_dict())
            if getattr(pretrained.model.decoder, "project_out", None) is not None and getattr(self, "project_out", None) is not None:
                self.project_out.load_state_dict(pretrained.model.decoder.project_out.state_dict())
            if pretrained.model.decoder.final_layer_norm is not None and self.final_norm is not None:
                self.final_norm.load_state_dict(pretrained.model.decoder.final_layer_norm.state_dict())
            src_blocks = pretrained.model.decoder.layers

        for i, block in enumerate(self.deq_blocks):
            if i < len(src_blocks):
                block.load_state_dict(src_blocks[i].state_dict())

        del pretrained
        torch.cuda.empty_cache()

        apply_norm(self.deq_blocks, args=self.deq_kwargs)
        apply_norm(self.input_inject, args=self.deq_kwargs)

    # ---- save / load ------------------------------------------------- #

    def save_pretrained(self, save_path, safe_serialization=False):
        os.makedirs(save_path, exist_ok=True)
        self.config.save_pretrained(save_path)
        meta = {
            "model_type": self.model_type,
            "n_deq_blocks": self.n_deq_blocks,
            "deq_kwargs": self.deq_kwargs,
        }
        with open(os.path.join(save_path, "deq_config.json"), "w") as f:
            json.dump(meta, f, indent=2, default=str)
        torch.save(self.state_dict(), os.path.join(save_path, "pytorch_model.bin"))

    @classmethod
    def from_pretrained(cls, load_path, device=None):
        config = AutoConfig.from_pretrained(load_path)
        with open(os.path.join(load_path, "deq_config.json")) as f:
            meta = json.load(f)
        model = cls(config, meta["model_type"], meta["deq_kwargs"], meta["n_deq_blocks"])
        sd = torch.load(os.path.join(load_path, "pytorch_model.bin"), map_location=device or "cpu")
        model.load_state_dict(sd)
        return model
