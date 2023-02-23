# pyright: reportPrivateImportUsage=false
from datasets import load_dataset


def collate_fn(tokenizer, examples):
    return tokenizer.pad(examples, padding="longest", return_tensors="pt")


def get_dataset(accelerator, tokenizer):
    data_files = {"train": "./data/train.csv", "test": "./data/test.csv"}
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
        processed_datasets = dataset.map(
            lambda examples: preprocess_function(tokenizer, examples),
            batched=True,
            num_proc=1,
            remove_columns=dataset["train"].column_names,
            load_from_cache_file=False,
            desc="Running tokenizer on dataset",
        )

    accelerator.wait_for_everyone()

    return processed_datasets


def preprocess_function(tokenizer, examples):
    posts = examples["post"]
    # posts = [clean(post) for post in posts]
    batch = tokenizer(posts, truncation=True, max_length=512, padding="max_length")

    # format [[0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 1, 0], ...]
    targets = [
        [float(round(examples[f"label_{i}"][row_index])) for i in range(7)]
        for row_index in range(len(posts))
    ]

    batch["labels"] = targets

    return batch
