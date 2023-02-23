import argparse

import evaluate
import torch
from accelerate import Accelerator
from accelerate.utils import find_executable_batch_size

# pyright: reportPrivateImportUsage=false
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
)


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


def main(args):
    accelerator = Accelerator()

    peft_config = LoraConfig(
        # the task type is hardcoded in this example to sequence classification
        task_type="SEQ_CLS",
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    # TODO: we currently always set the seed which might not be required in all cases
    set_seed(args.seed)

    # creating dataloaders
    # train_dataloader, eval_dataloader = get_dataloaders(
    #     accelerator=accelerator,
    #     model_name=args.model_name,
    #     batch_size=args.batch_size,
    #     seq_len=args.seq_len,
    # )

    # we do set the seed before to also control weight initialization
    model = AutoModelForSequenceClassification.from_pretrained(
        # TODO: this is hardcoded the the number of labels in the Stackoverflow dataset
        # and the fact that it is a multi-label classification task
        args.model_name,
        num_labels=7,
        problem_type="multi_label_classification",
    )
    model = get_peft_model(model, peft_config)
    # prints some nice information about the trainable parameters
    # and model stastics
    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # lr_scheduler = get_linear_schedule_with_warmup(
    #     optimizer=optimizer,
    #     num_warmup_steps=0.06 * (len(train_dataloader) * args.num_epochs),
    #     num_training_steps=(len(train_dataloader) * args.num_epochs),
    # )

    # accelerate wraps the model, optimizer, etc to allow for distributed training
    # and other desired functionalities
    # model, train_dataloader, eval_dataloader, optimizer, lr_scheduler = accelerator.prepare(
    #     model,
    #     train_dataloader,
    #     eval_dataloader,
    #     optimizer,
    #     lr_scheduler,
    # )

    model, optimizer = accelerator.prepare(model, optimizer)

    starting_epoch = 0
    # we can combine multiple evaluation metrics
    # see hf evaluate for more details
    eval_metrics = evaluate.load("f1", "multilabel")

    # TODO: if args.resume_from_checkpoint is not None:
    # load checkpoint
    # load epoch, ...

    # we wrap the inner training loop into a function
    # in order to find the maximum batch size that fits into memory
    @find_executable_batch_size(starting_batch_size=args.batch_size)
    def training_loop(batch_size: int):
        nonlocal accelerator  # ensure they can be used in our context
        accelerator.free_memory()  # free lingering references

        train_dataloader, eval_dataloader = get_dataloaders(
            accelerator=accelerator,
            model_name=args.model_name,
            batch_size=batch_size,
            seq_len=args.seq_len,
        )

        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=0.06 * (len(train_dataloader) * args.num_epochs),
            num_training_steps=(len(train_dataloader) * args.num_epochs),
        )

        train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
            train_dataloader, eval_dataloader, lr_scheduler
        )

        # investigate if this is necessary
        # https://github.com/huggingface/accelerate/blob/main/examples/by_feature/automatic_gradient_accumulation.py
        gradient_accumulation_steps = 1
        if batch_size < args.batch_size:
            gradient_accumulation_steps = args.batch_size // batch_size
            accelerator.print(
                f"Batch size {args.batch_size} is too large for memory, reducing to {batch_size} and accumulating gradients over {gradient_accumulation_steps} steps"
            )

        for epoch in range(starting_epoch, args.num_epochs):
            model.train()
            total_loss = 0

            # if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            #     # We need to skip steps until we reach the resumed step
            #     train_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
            #     overall_step += resume_step

            for step, batch in enumerate(train_dataloader):
                outputs = model(**batch)
                loss = outputs.loss / gradient_accumulation_steps
                total_loss += loss.detach().float()
                accelerator.backward(loss)
                if step % gradient_accumulation_steps == 0:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                if step % args.print_freq == 0:
                    accelerator.print(
                        f"[Training] Epoch: {epoch} | Step {step}/{len(train_dataloader)} - Loss: {loss:.2f} - LR: {optimizer.param_groups[0]['lr']:.7f}"
                    )

            accelerator.print(
                f"[Training] Epoch: {epoch} Completed | Loss: {(total_loss / (len(train_dataloader))):.2f} - LR: {optimizer.param_groups[0]['lr']:.7f}"
            )

            model.eval()
            total_loss = 0

            for step, batch in enumerate(eval_dataloader):
                with torch.no_grad():
                    outputs = accelerator.unwrap_model(model)(**batch)
                loss = outputs.loss / gradient_accumulation_steps
                total_loss += loss.detach().float()
                # we just threshold at .5, not optimal but good enough for a sample script
                predictions = (torch.sigmoid(outputs.logits) > 0.5) * 1
                # we need to gather the predictions and references on all processes
                predictions, references = accelerator.gather_for_metrics(
                    (predictions, batch["labels"])
                )
                eval_metrics.add_batch(
                    predictions=predictions.tolist(),
                    references=references.int().tolist(),
                )

                if step % args.print_freq == 0:
                    accelerator.print(
                        f"[Eval] Epoch: {epoch} | Step {step}/{len(eval_dataloader)} - Loss: {loss:.2f}"
                    )

            metrics: dict = eval_metrics.compute(average="weighted")

            accelerator.print(
                f"[Eval] Epoch: {epoch} Completed | Loss: {(total_loss / (len(eval_dataloader))):.2f} - F1: {metrics['f1']:.2f}"
            )

    # batch_size is not needed here, as the decorator will set it
    # pylint: disable=no-value-for-parameter
    training_loop()

    #     if args.with_tracking:
    #         accelerator.log(
    #             {
    #                 "accuracy": eval_metric["accuracy"],
    #                 "f1": eval_metric["f1"],
    #                 "train_loss": total_loss.item() / len(train_dataloader),
    #                 "epoch": epoch,
    #             },
    #             step=epoch,
    #         )

    #         output_dir = f"epoch_{epoch}"
    #         if args.output_dir is not None:
    #             output_dir = os.path.join(args.output_dir, output_dir)
    #         accelerator.save_state(output_dir)

    accelerator.end_training()
    accelerator.wait_for_everyone()


def parse_args():
    parser = argparse.ArgumentParser(description="Multilabel text classification")

    parser.add_argument("--model-name", default="roberta-large", type=str)

    parser.add_argument(
        "--batch-size",
        default=8,
        type=int,
        help="the training will find the max batch size that fits into memory and use gradient accumulation",
    )
    parser.add_argument(
        "--num-epochs", default=5, type=int, metavar="N", help="number of total epochs to run"
    )
    parser.add_argument("--seq-len", default=512, type=int, help="sequence length")
    parser.add_argument("--learning-rate", default=0.0003, type=float, help="initial learning rate")
    parser.add_argument("--seed", default=42, type=int, help="seed for initializing training. ")

    # LoRa
    parser.add_argument("--lora-r", default=8, type=int, help="LoRa r")
    parser.add_argument("--lora-alpha", default=16, type=int, help="LoRa alpha")
    parser.add_argument("--lora-dropout", default=0.1, type=float, help="LoRa dropout")

    parser.add_argument("--print-freq", default=10, type=int, help="print frequency")
    parser.add_argument("--dataset-dir", default="./dataset", help="dataset path")
    parser.add_argument("--output-dir", default=".", help="path where to save")
    parser.add_argument("--resume", default="", help="resume from checkpoint")

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
