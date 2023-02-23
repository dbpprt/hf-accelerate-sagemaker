import argparse

import evaluate
import torch
from accelerate import Accelerator
# pyright: reportPrivateImportUsage=false
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup, set_seed)


def get_dataloaders(accelerator: Accelerator, model_name: str, batch_size: int, seq_len: int):
    """
    Creates a set of `DataLoader`s for a sample Stackoverflow dataset (CSV)

    Args:
        accelerator (`Accelerator`):
            An accelerator object.
        model_name (`str`):
            The name of the model to use.
        batch_size (`int`):
            The batch size to use for both training and evaluation.
        seq_len (`int`):
            The maximum sequence length to use for both training and evaluation.
    """

    # some boilerplate required to work with different model architectures
    # TODO: really required?
    if any(k in model_name for k in ("gpt", "opt", "bloom")):
        padding_side = "left"
    else:
        padding_side = "right"

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side=padding_side)
    if getattr(tokenizer, "pad_token_id") is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    def collate_fn(examples):
        return tokenizer.pad(examples, padding="longest", return_tensors="pt")

    def preprocess_function(examples: dict):
        """
        Preprocesses the given dataset examples by tokenizing them and truncating them to the maximum sequence length.
        We also gather labels specific to the Stackoverflow dataset.

        Args:
            examples (`dict`):
                The dataset examples to preprocess.
        """
        batch = tokenizer(
            examples["post"], truncation=True, max_length=seq_len, padding="max_length"
        )
        # format [[0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 1, 0], ...]
        batch["labels"] = [
            [float(round(examples[f"label_{i}"][row_index])) for i in range(7)]
            for row_index in range(len(examples["post"]))
        ]

        return batch

    data_files = {"train": "./data/train.csv", "test": "./data/test.csv"}

    # this is really specific to the Stackoverflow dataset
    # change it for your respective dataset
    dataset = load_dataset(
        "csv",
        data_files=data_files,
        column_names=[
            "label_0",
            "label_1",
            "label_2",
            "label_3",
            "label_4",
            "label_5",
            "label_6",
            "post",
        ],
    )

    with accelerator.main_process_first():
        dataset = dataset.map(
            preprocess_function,
            batched=True,
            num_proc=1,
            remove_columns=dataset["train"].column_names,
            # ideally we should use the cache, but this is a sample script, hence we don't
            load_from_cache_file=False,
            desc="Running tokenizer on dataset",
        )

    accelerator.wait_for_everyone()

    train_dataloader = DataLoader(
        dataset["train"],
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=batch_size,
        pin_memory=True,
    )

    eval_dataloader = DataLoader(
        dataset["test"], collate_fn=collate_fn, batch_size=batch_size, pin_memory=True
    )

    return train_dataloader, eval_dataloader