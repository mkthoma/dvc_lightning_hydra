import os
import pytest
from hydra import initialize, compose
import rootutils
from typing import List
from lightning.pytorch.loggers import Logger
import lightning as L
import hydra
import torch

# Setup root directory
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.train import run_train, run_test, instantiate_callbacks, instantiate_loggers
from src.datamodules.dogbreed import DogBreedImageDataModule
from src.models.timm_classifier import TimmClassifier

def test_train_and_test():
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="train.yaml", overrides=["test=False", "trainer.fast_dev_run=True"])

    # Override paths in the config
    project_root = rootutils.find_root(indicator=".project-root")
    cfg.paths.root_dir = str(project_root)
    cfg.paths.data_dir = str(project_root / "data")
    cfg.paths.log_dir = str(project_root / "logs")
    cfg.paths.output_dir = os.path.join(cfg.paths.log_dir, "pytest")
    cfg.paths.work_dir = str(project_root)

    # Create the output directory if it doesn't exist
    os.makedirs(cfg.paths.output_dir, exist_ok=True)

    # Data preparation
    datamodule: DogBreedImageDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.prepare_data()
    datamodule.setup()

    model: TimmClassifier = hydra.utils.instantiate(cfg.model)

    # Set up callbacks
    callbacks: List[L.Callback] = instantiate_callbacks(cfg.get("callbacks"))

    # Set up loggers
    loggers: List[Logger] = instantiate_loggers(cfg.get("logger"))

    # Initialize Trainer
    trainer: L.Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=loggers,
    )

    # Run the training function
    train_metrics = run_train(cfg, trainer, model, datamodule)
    print("Train_metrics: ", train_metrics)
    assert "train/loss" in train_metrics, "Train loss should be in metrics"
    assert "train/acc" in train_metrics, "Train accuracy should be in metrics"

    # Run the test function
    test_metrics = run_test(cfg, trainer, model, datamodule)
    print("Test_metrics: ", test_metrics)
    assert isinstance(test_metrics, list), "test should return a list"
    assert len(test_metrics) > 0, "Test metrics should not be empty"
    
    test_results = test_metrics[0]
    assert "test/loss" in test_results, "Test loss should be in metrics"
    assert "test/acc" in test_results, "Test accuracy should be in metrics"

    # Check if metric values are within expected ranges
    assert 0 <= train_metrics["train/loss"] <= 10, "Train loss should be between 0 and 10"
    assert 0 <= train_metrics["train/acc"] <= 1, "Train accuracy should be between 0 and 1"
    assert 0 <= test_results["test/loss"] <= 10, "Test loss should be between 0 and 10"
    assert 0 <= test_results["test/acc"] <= 1, "Test accuracy should be between 0 and 1"

    # Check if test results are saved
    results_file = os.path.join(cfg.paths.output_dir, "test_results.pt")
    assert os.path.exists(results_file), "Test results file should exist"
    
    # Load and verify saved test results
    saved_results = torch.load(results_file)
    assert saved_results == test_results, "Saved test results should match the returned test metrics"