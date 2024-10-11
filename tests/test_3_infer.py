import os
import pytest
from hydra import initialize, compose
import rootutils
from typing import List
from lightning.pytorch.loggers import Logger
import lightning as L
import hydra
import torch
import glob
from PIL import Image

# Setup root directory
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
print(f"Project root: {root}")

from src.infer import infer, instantiate_callbacks, instantiate_loggers
from src.models.timm_classifier import TimmClassifier

def test_infer():
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="infer.yaml")

    # Override paths in the config
    project_root = rootutils.find_root(indicator=".project-root")
    cfg.paths.root_dir = str(project_root)
    cfg.paths.log_dir = str(project_root / "logs")
    cfg.paths.output_dir = os.path.join(cfg.paths.log_dir, "pytest_infer")
    cfg.paths.work_dir = str(project_root)
    cfg.paths.model_artifacts = os.path.join(cfg.paths.output_dir, "model_artifacts")

    # Create the output and input directories
    os.makedirs(cfg.paths.output_dir, exist_ok=True)
    os.makedirs(os.path.join(cfg.paths.model_artifacts, "input_images"), exist_ok=True)

    # Create dummy input images
    input_dir = os.path.join(cfg.paths.model_artifacts, "input_images")
    dummy_image = Image.new('RGB', (224, 224), color='red')
    dummy_image.save(os.path.join(input_dir, "dummy_image.jpg"))

    # Set up model
    model: TimmClassifier = hydra.utils.instantiate(cfg.model)

    # Create a dummy checkpoint
    dummy_checkpoint = os.path.join(cfg.paths.output_dir, "dummy_checkpoint.ckpt")
    torch.save({"state_dict": model.state_dict()}, dummy_checkpoint)

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

    # Run inference
    try:
        infer(cfg)
    except Exception as e:
        pytest.fail(f"Inference failed with error: {str(e)}")

    # Check if output directory is created
    output_dir = os.path.join(cfg.paths.model_artifacts, "predicted_images")
    assert os.path.exists(output_dir), "Output directory should exist"

    # Check if predicted image is created
    predicted_images = glob.glob(os.path.join(output_dir, "pred_*.jpg"))
    assert len(predicted_images) > 0, "At least one predicted image should be created"

