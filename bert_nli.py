import io
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer
import numpy as np
from datasets import load_metric
from transformers import AdamW
from torch.utils.data import DataLoader
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
from torch.utils.data import Dataset


mnli_train_fpath = '/home/hurunyi/MNLI/train.tsv'
mnli_dev_fpath = '/home/hurunyi/MNLI/dev_matched.tsv'
checkpoint_dir = '/home/hurunyi/bert_score/bert_nli_outputs/checkpoint-8000'

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


class mnli_dataset(Dataset):
    def __init__(self, fpath):
        label2id = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
        self.sent1 = [line.split("\t")[8] for line in io.open(fpath, encoding='utf8').read().splitlines()[1:]]
        self.sent2 = [line.split("\t")[9] for line in io.open(fpath, encoding='utf8').read().splitlines()[1:]]
        self.gs_scores = np.array([label2id[line.split("\t")[11]] \
                               for line in io.open(fpath, encoding='utf8').read().splitlines()[1:]])

    def __len__(self):
        return len(self.gs_scores)

    def __getitem__(self, idx):
        encoded_inputs = tokenizer(self.sent1[idx], self.sent2[idx], padding="max_length", truncation=True)

        encoded_inputs["attention_mask"] = torch.tensor(encoded_inputs["attention_mask"])
        encoded_inputs["input_ids"] = torch.tensor(encoded_inputs["input_ids"])
        encoded_inputs["labels"] = torch.tensor(self.gs_scores[idx])
        encoded_inputs["token_type_ids"] = torch.tensor(encoded_inputs["token_type_ids"])

        return encoded_inputs


def my_collate_fn(batch):
    result = {}
    attention_mask, input_ids, labels, token_type_ids = [], [], [], []

    for sent in batch:
        if sent["labels"].item() != -1:
            attention_mask.append(sent["attention_mask"])
            input_ids.append(sent["input_ids"])
            labels.append(sent["labels"])
            token_type_ids.append(sent["token_type_ids"])

    result["attention_mask"] = torch.stack(attention_mask)
    result["input_ids"] = torch.stack(input_ids)
    result["labels"] = torch.stack(labels)
    result["token_type_ids"] = torch.stack(token_type_ids)

    return result


def compute_metrics(eval_pred):
    metric = load_metric("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def tokenize_function(examples):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    return tokenizer(examples["premise"], examples["hypothesis"], padding="max_length", truncation=True)


def snli():
    raw_datasets = load_dataset("'multi_nli'")
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["premise", "hypothesis"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"]
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)

    training_args = TrainingArguments("bert_nli_outputs", fp16=True,
                                      per_device_train_batch_size=24, per_device_eval_batch_size=24)
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=my_collate_fn,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()


def mnli():
    train_dataset = mnli_dataset(mnli_train_fpath)
    eval_dataset = mnli_dataset(mnli_dev_fpath)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir, num_labels=3)
    training_args = TrainingArguments("bert_mnli_outputs", fp16=True,
                                      per_device_train_batch_size=24, per_device_eval_batch_size=24)
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=my_collate_fn,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()


if __name__ == "__main__":
    snli()
    mnli()
