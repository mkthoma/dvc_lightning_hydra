import os
from pathlib import Path
import logging

import hydra
from omegaconf import DictConfig
import lightning as L
from lightning.pytorch.loggers import Logger
from typing import List
import torch

import rootutils

# Setup root directory
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
print(f"Project root: {root}")

# Imports that require root directory setup
from src.utils import logging_utils
from src.datamodules.dogbreed import DogBreedImageDataModule
from src.models.timm_classifier import TimmClassifier

log = logging.getLogger(__name__)

def instantiate_callbacks(callback_cfg: DictConfig) -> List[L.Callback]:
    callbacks: List[L.Callback] = []
    if not callback_cfg:
        log.warning("No callback configs found! Skipping..")
        return callbacks

    for _, cb_conf in callback_cfg.items():
        if "_target_" in cb_conf:
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks

def instantiate_loggers(logger_cfg: DictConfig) -> List[Logger]:
    loggers: List[Logger] = []
    if not logger_cfg:
        log.warning("No logger configs found! Skipping..")
        return loggers

    for _, lg_conf in logger_cfg.items():
        if "_target_" in lg_conf:
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            loggers.append(hydra.utils.instantiate(lg_conf))

    return loggers

@logging_utils.task_wrapper
def run_train(cfg: DictConfig, trainer: L.Trainer, model: TimmClassifier, datamodule: DogBreedImageDataModule):
    log.info("Starting training!")
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))
    
    train_metrics = trainer.callback_metrics
    log.info(f"Training metrics:\n{train_metrics}")
    return train_metrics
@logging_utils.task_wrapper
def run_test(cfg: DictConfig, trainer: L.Trainer, model: TimmClassifier, datamodule: DogBreedImageDataModule):
    log.info("Starting testing!")
    
    if trainer.checkpoint_callback.best_model_path:
        log.info(f"Loading best checkpoint: {trainer.checkpoint_callback.best_model_path}")
        test_metrics = trainer.test(model=model, datamodule=datamodule, ckpt_path=trainer.checkpoint_callback.best_model_path)
    else:
        log.warning("No checkpoint found! Using current model weights.")
        test_metrics = trainer.test(model=model, datamodule=datamodule)
    
    log.info(f"Test metrics:\n{test_metrics}")

    # Save test results
    results_file = os.path.join(cfg.paths.output_dir, "test_results.pt")
    torch.save(test_metrics[0], results_file)
    log.info(f"Test results saved to {results_file}")
    return test_metrics

@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    # Setup logger for the entire script
    logging_utils.setup_logger(log_file=f"{cfg.paths.output_dir}/train.log")
    
    # Initialize DataModule
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: DogBreedImageDataModule = hydra.utils.instantiate(cfg.data)

    # Initialize Model
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: TimmClassifier = hydra.utils.instantiate(cfg.model)

    # Set up callbacks
    callbacks: List[L.Callback] = instantiate_callbacks(cfg.get("callbacks"))

    # Set up loggers
    loggers: List[Logger] = instantiate_loggers(cfg.get("logger"))

    # Initialize Trainer
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: L.Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=loggers,
    )

    # Train the model
    if cfg.get("train", True):
        run_train(cfg, trainer, model, datamodule)

    # Test the model
    if cfg.get("test", False):
        run_test(cfg, trainer, model, datamodule)

if __name__ == "__main__":
    main()