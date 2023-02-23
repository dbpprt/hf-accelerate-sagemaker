import argparse

from accelerate import Accelerator

# pyright: reportPrivateImportUsage=false
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


def get_dataloaders(
    accelerator: Accelerator, model_name: str, data_dir: str, batch_size: int, seq_len: int
):
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

    dataset = load_from_disk(data_dir)

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


def main(args):
    data_files = {"train": args.train_file, "test": args.test_file}

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

    dataset.save_to_disk(args.output_dir, storage_options={"anon": True})
    # see known issue: https://github.com/huggingface/datasets/issues/5281
    # builder.download_and_prepare(
    #     args.output_dir, storage_options={"anon": True}, file_format=args.file_format
    # )


def parse_args():
    parser = argparse.ArgumentParser(description="Data preprocessing script")
    parser.add_argument("--train_file", type=str, default="./data/train.csv")
    parser.add_argument("--test_file", type=str, default="./data/test.csv")
    parser.add_argument("--output_dir", type=str, default="./.data")
    parser.add_argument("--file_format", type=str, default="parquet")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
