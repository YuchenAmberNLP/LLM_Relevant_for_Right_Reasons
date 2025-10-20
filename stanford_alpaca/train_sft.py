# -*- coding: utf-8 -*-
"""
This script is modified from:
https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py

Original project: Stanford Alpaca (Tatsu Lab)
Original file: train.py

This version has been adapted and extended by Yuchen Mao for her Master's thesis project:
"Relevant for the Right Reasons? Investigating Lexical Biases in Re-ranking with Large Language Models".


For academic use only. See the original Stanford Alpaca repository for license and usage terms.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer

import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import transformers
import utils
# from torch.utils.data import Dataset
from transformers import Trainer
from datasets import Dataset, load_dataset

from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM



IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="meta-llama/Llama-3.1-8B-Instruct")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
# class TrainingArguments(transformers.TrainingArguments):
#     cache_dir: Optional[str] = field(default=None)
#     optim: str = field(default="adamw_torch")
#     model_max_length: int = field(
#         default=512,
#         metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
#     )
class TrainingArguments(SFTConfig):
    packing=False
    dataset_text_field="text"
    output_dir: str = field(default="../model_output")
    cache_dir: Optional[str] = field(default=None)
    model_max_length: int = field(default=1024)
    optim: str = field(default="adamw_torch")
    completion_only_loss=True
    bf16: bool = field(default=True)
    tf32: bool = field(default=True)
    num_train_epochs: int = field(default=1)
    per_device_train_batch_size: int = field(default=2)
    per_device_eval_batch_size: int = field(default=2)
    gradient_accumulation_steps: int = field(default=16)
    gradient_checkpointing: bool = field(default=True)

    save_strategy: str = field(default="steps")
    save_steps: int = field(default=2000)
    save_total_limit: int = field(default=1)
    logging_steps: int = field(default=100)

    learning_rate: float = field(default=2e-5)
    weight_decay: float = field(default=0.0)
    warmup_ratio: float = field(default=0.03)

    deepspeed: Optional[str] = field(default="./configs/default_offload_opt_param.json")


def smart_tokenizer_and_embedding_resize(
        special_tokens_dict: Dict,
        tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def build_messages_dataset(data_path: str, tokenizer) -> Dataset:
    """Build a messages-formatted dataset for SFTTrainer-compatible chat fine-tuning."""
    logging.warning("Loading data...")
    list_data_dict = utils.jload(data_path)

    logging.warning("Formatting inputs...")

    messages_dataset = []
    for example in list_data_dict:
        system_prompt = example["instruction"]
        user_input = example["input"]
        response = example["output"]

        message = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": response}
        ]
        formatted_message = tokenizer.apply_chat_template(message, tokenize=False) + tokenizer.eos_token
        messages_dataset.append({"text": formatted_message})

    logging.warning(f"Formatted {len(messages_dataset)} samples.")
    return Dataset.from_list(messages_dataset)


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    # tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.model_name_or_path)


    tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
    )

    response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    collator = DataCollatorForCompletionOnlyLM(response_template=response_template, tokenizer=tokenizer, mlm=False)


# model, tokenizer = setup_chat_format(model, tokenizer) -  not needed for llama3.2 instruct
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )
    dataset = build_messages_dataset(data_args.data_path, tokenizer)
    # change to use SFTTrainer
    # change data format
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        data_collator=collator,
        args=training_args,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()