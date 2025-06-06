import os
import mlflow
import torch
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import MLFlowLogger
from omegaconf import DictConfig, OmegaConf
import hydra

from .data import TextGenerationDataModule
from .model import TextGenerationClassifier

def train(cfg: DictConfig) -> None:
    # Initialize MLflow
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.experiment.name)
    mlflow.start_run()
    mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))
    
    # Resolve file paths
    train_file = hydra.utils.to_absolute_path(cfg.data.train_file)
    val_file = hydra.utils.to_absolute_path(cfg.data.val_file)
    
    # Data module
    data_module = TextGenerationDataModule(
        train_file=train_file,
        val_file=val_file,
        tokenizer_name=cfg.model.name,
        batch_size=cfg.data.batch_size,
        max_length=cfg.data.max_length
    )
    
    # Calculate total training steps
    train_data = pd.read_parquet(train_file)
    num_training_steps = (len(train_data) // cfg.data.batch_size) * cfg.training.epochs
    
    # Model
    model = TextGenerationClassifier(cfg, num_training_steps)
    
    # MLflow Logger
    mlflow_logger = MLFlowLogger(
        experiment_name=cfg.experiment.name,
        tracking_uri=cfg.mlflow.tracking_uri,
        run_id=mlflow.active_run().info.run_id
    )
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor=cfg.callbacks.checkpoint.monitor,
        mode=cfg.callbacks.checkpoint.mode,
        dirpath=cfg.callbacks.checkpoint.dirpath,
        filename=cfg.callbacks.checkpoint.filename,
        save_top_k=1
    )
    
    early_stopping_callback = EarlyStopping(
        monitor=cfg.callbacks.early_stopping.monitor,
        patience=cfg.callbacks.early_stopping.patience,
        mode=cfg.callbacks.early_stopping.mode
    )
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.training.epochs,
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=mlflow_logger,
        accelerator='auto',
        devices='auto',
        log_every_n_steps=cfg.training.log_every_n_steps,
        enable_progress_bar=not cfg.training.disable_progress_bar,
        limit_train_batches=cfg.training.limit_train_batches,
        limit_val_batches=cfg.training.limit_val_batches
    )
    
    # Train
    trainer.fit(model, data_module)
    
    # Log best model
    mlflow.pytorch.log_model(
        pytorch_model=model,
        artifact_path="model",
        registered_model_name=cfg.mlflow.registered_model_name
    )
    
    # Log best checkpoint
    mlflow.log_artifact(
        local_path=checkpoint_callback.best_model_path,
        artifact_path="checkpoints"
    )
    
    mlflow.end_run()
    