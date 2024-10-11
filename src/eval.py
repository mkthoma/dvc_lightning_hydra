import os
import hydra
from omegaconf import DictConfig
import lightning as L
from lightning.pytorch.loggers import Logger
from typing import List
import glob
import rootutils

# Setup root directory
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
print(f"Project root: {root}")

# Imports that require root directory setup
from src.utils import logging_utils
from src.datamodules.dogbreed import DogBreedImageDataModule
from src.models.timm_classifier import TimmClassifier

log = logging_utils.logger

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
def evaluate(cfg: DictConfig):
    # Set up data module
    log.info("Instantiating datamodule")
    datamodule: DogBreedImageDataModule = hydra.utils.instantiate(cfg.data)
    
    # Prepare data and setup
    datamodule.prepare_data()
    datamodule.setup(stage="fit")
    
    # Ensure validation dataset is initialized correctly
    if datamodule.val_dataset is None:
        raise ValueError("Validation dataset is not initialized. Check the setup method.")

    # Get the number of classes from the data module
    num_classes = len(datamodule.val_dataset.dataset.classes)
    log.info(f"Number of classes detected: {num_classes}")
    
    # Set up callbacks
    callbacks: List[L.Callback] = instantiate_callbacks(cfg.get("callbacks"))

    # Set up logger
    loggers: List[Logger] = instantiate_loggers(cfg.get("logger"))

    # Set up trainer
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: L.Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=loggers,
    )

    # Load the checkpoint with the best validation accuracy
    runs_dir = os.path.join(cfg.paths.log_dir, cfg.train_task_name, "runs")
    log.info(f"Runs directory: {runs_dir}")

    checkpoints = glob.glob(os.path.join(runs_dir, "**", "*.ckpt"), recursive=True)

    if checkpoints:
        best_checkpoint = max(checkpoints, key=os.path.getmtime)
        log.info(f"Loading best checkpoint: {best_checkpoint}")
        results = trainer.validate(model=TimmClassifier.load_from_checkpoint(best_checkpoint), datamodule=datamodule)
    else:
        log.warning("No checkpoints found! Using initialized model weights.")
        log.info(f"Original model config: {cfg.model}")  # Log the original config
        model_cfg = {k: v for k, v in cfg.model.items() if k != '_target_'}
        log.info(f"Filtered model config: {model_cfg}")  # Log the filtered config
        model = TimmClassifier(**model_cfg)
        results = trainer.validate(model=model, datamodule=datamodule)

    # Print validation metrics
    log.info("Validation Metrics:")
    for k, v in results[0].items():
        log.info(f"{k}: {v}")

    # Print callback metrics
    log.info("Callback Metrics:")
    callback_metrics = {}
    for k, v in trainer.callback_metrics.items():
        log.info(f"{k}: {v}")
        callback_metrics[k] = v

    log.info("Evaluation complete")

    # Return both validation and callback metrics
    return {
        "validation_metrics": results[0],
        "callback_metrics": callback_metrics
    }

@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    # Setup logger for the entire script
    logging_utils.setup_logger(log_file=f"{cfg.paths.output_dir}/eval.log")
    
    # Evaluate the model
    evaluation_metrics = evaluate(cfg)

    # Print evaluation metrics
    log.info("Evaluation Metrics:")
    for k, v in evaluation_metrics.items():
        log.info(f"{k}: {v}")

if __name__ == "__main__":
    main()
