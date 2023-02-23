import torch
from accelerate import Accelerator

# pyright: reportPrivateImportUsage=false
# from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
)

from stackoverflow import get_dataset


def main():
    accelerator = Accelerator()
    model_name_or_path = "bigscience/mt0-base"

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
    )

    learning_rate = 3e-3
    num_epochs = 10
    batch_size = 16
    seed = 42

    set_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    processed_datasets, targets = get_dataset(accelerator=accelerator, tokenizer=tokenizer)

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["test"]

    def collate_fn(examples):
        return tokenizer.pad(examples, padding="longest", return_tensors="pt")

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=batch_size,
        pin_memory=True,
    )
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=collate_fn, batch_size=batch_size, pin_memory=True
    )

    # creating model
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # lr scheduler
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_dataloader) * num_epochs),
    )

    (
        model,
        train_dataloader,
        eval_dataloader,
        optimizer,
        lr_scheduler,
    ) = accelerator.prepare(
        model,
        train_dataloader,
        eval_dataloader,
        optimizer,
        lr_scheduler,
    )

    is_ds_zero_3 = False
    if getattr(accelerator.state, "deepspeed_plugin", None):
        # pyright: reportOptionalMemberAccess=false
        is_ds_zero_3 = accelerator.state.deepspeed_plugin.zero_stage == 3

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for _, batch in enumerate(tqdm(train_dataloader)):
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.detach().float()
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        train_epoch_loss = total_loss / len(eval_dataloader)
        train_ppl = torch.exp(train_epoch_loss)
        accelerator.print(f"{epoch=}: {train_ppl=} {train_epoch_loss=}")

        model.eval()
        eval_preds = []

        for _, batch in enumerate(tqdm(eval_dataloader)):
            batch = {k: v for k, v in batch.items() if k != "labels"}
            with torch.no_grad():
                outputs = accelerator.unwrap_model(model).generate(
                    **batch, synced_gpus=is_ds_zero_3
                )  # synced_gpus=True for DS-stage 3
            preds = outputs.detach().cpu().numpy()
            eval_preds.extend(tokenizer.batch_decode(preds, skip_special_tokens=True))

        correct = 0
        total = 0
        for pred, true in zip(eval_preds, targets):
            if pred.strip() == true.strip():
                correct += 1
            total += 1
        accuracy = correct / total * 100
        accelerator.print(f"{accuracy=}")
        accelerator.print(f"{eval_preds[:10]=}")

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
