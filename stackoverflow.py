# pyright: reportPrivateImportUsage=false
from datasets import load_dataset

label_mappings = {
    0: "Analytics and AI/ML",
    1: "Networking",
    2: "Governance and Security",
    3: "Database and Storage",
    4: "Compute",
    5: "Application Development, Application Integration and Ops",
    6: "Other",
}


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

        targets = processed_datasets["test"]["targets"]
        processed_datasets = processed_datasets.remove_columns("targets")
    accelerator.wait_for_everyone()

    return processed_datasets, targets


import html
import warnings

import nltk

nltk.download("punkt")

label_mappings = {
    0: "Analytics and AI/ML",
    1: "Networking",
    2: "Governance and Security",
    3: "Database and Storage",
    4: "Compute",
    5: "Application Development, Application Integration and Ops",
    6: "Other",
}


def tokenize(text):
    return nltk.word_tokenize(text.lower())


def remove_html_tags(text):
    """Remove html tags from a string"""
    import re

    clean = re.compile("<.*?>")
    return re.sub(clean, "", text)


def clean(text):
    result = html.unescape(text)
    result = remove_html_tags(result)
    # result = tokenize(text)
    return result


def preprocess_function(tokenizer, examples):
    posts = examples["post"]
    posts = [clean(post) for post in posts]
    batch = tokenizer(posts, truncation=True, max_length=512, padding="max_length")

    # format [[0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 1, 0], ...]
    targets = [
        [round(examples[f"label_{i}"][row_index]) for i in range(7)]
        for row_index in range(len(posts))
    ]

    # targets = ["".join([str(i) for i in target]) for target in targets]
    targets = [" ".join(["yes" if i == 1 else "no" for i in target]) for target in targets]
    max_len = max([len(tokenizer(target)["input_ids"]) for target in targets])

    batch["targets"] = targets
    batch["labels"] = tokenizer(
        targets,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )["input_ids"]
    batch["labels"][batch["labels"] == tokenizer.pad_token_id] = -100

    return batch
