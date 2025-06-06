from typing import Optional

import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


class TextGenerationDataset(Dataset):
    def __init__(self, filepath: str, tokenizer: AutoTokenizer, max_length: int = 128):
        self.data = pd.read_parquet(filepath)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = str(self.data.iloc[idx]["text"])
        label = int(self.data.iloc[idx]["generated"])

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


class TextGenerationDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_file: str,
        val_file: str,
        tokenizer_name: str,
        batch_size: int = 32,
        max_length: int = 128,
    ):
        super().__init__()
        self.train_file = train_file
        self.val_file = val_file
        self.tokenizer_name = tokenizer_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.batch_size = batch_size
        self.max_length = max_length

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = TextGenerationDataset(
            self.train_file, self.tokenizer, self.max_length
        )
        self.val_dataset = TextGenerationDataset(
            self.val_file, self.tokenizer, self.max_length
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4
        )
