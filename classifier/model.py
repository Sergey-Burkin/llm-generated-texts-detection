from typing import Dict, List

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import AdamW
from transformers import (
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)


class TextGenerationClassifier(pl.LightningModule):
    def __init__(self, cfg: DictConfig, num_training_steps: int):
        super().__init__()
        self.cfg = cfg
        self.model = AutoModelForSequenceClassification.from_pretrained(
            cfg.model.name, num_labels=2
        )
        self.num_training_steps = num_training_steps
        self.validation_outputs: List[Dict[str, torch.Tensor]] = []

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

    def training_step(self, batch, batch_idx):
        outputs = self(batch["input_ids"], batch["attention_mask"], batch["labels"])
        self.log("train_loss", outputs.loss, prog_bar=True)
        return outputs.loss

    def validation_step(self, batch, batch_idx):
        outputs = self(batch["input_ids"], batch["attention_mask"], batch["labels"])
        loss = outputs.loss
        preds = torch.argmax(outputs.logits, dim=1)

        self.log("val_loss", loss, prog_bar=True)
        self.validation_outputs.append({"preds": preds, "labels": batch["labels"]})
        return loss

    def on_validation_epoch_end(self):
        preds = torch.cat([x["preds"] for x in self.validation_outputs])
        labels = torch.cat([x["labels"] for x in self.validation_outputs])

        val_acc = accuracy_score(labels.cpu(), preds.cpu())
        val_f1 = f1_score(labels.cpu(), preds.cpu())

        self.log("val_acc", val_acc, prog_bar=True)
        self.log("val_f1", val_f1, prog_bar=True)
        self.validation_outputs.clear()

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.cfg.training.learning_rate,
            weight_decay=self.cfg.training.weight_decay,
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.cfg.training.warmup_steps,
            num_training_steps=self.num_training_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
