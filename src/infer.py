import os
import hydra
from omegaconf import DictConfig
import lightning as L
import torch
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import glob
import rootutils
from typing import List
from lightning.pytorch.loggers import Logger

# Setup root directory
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
print(f"Project root: {root}")

# Imports that require root directory setup
from src.utils import logging_utils
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
def infer(cfg: DictConfig):
    # Set up callbacks
    callbacks: List[L.Callback] = instantiate_callbacks(cfg.get("callbacks"))

    # Set up loggers
    loggers: List[Logger] = instantiate_loggers(cfg.get("logger"))

    # Set up trainer
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: L.Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=loggers,
    )

    # Set up model
    log.info("Instantiating model")
    model: TimmClassifier = hydra.utils.instantiate(cfg.model)

    # Load the best checkpoint
    runs_dir = os.path.join(cfg.paths.log_dir, cfg.train_task_name, "runs")
    log.info(f"Runs directory: {runs_dir}")

    checkpoints = glob.glob(os.path.join(runs_dir, "**", "*.ckpt"), recursive=True)

    if checkpoints:
        best_checkpoint = max(checkpoints, key=os.path.getmtime)
        log.info(f"Loading best checkpoint: {best_checkpoint}")
        model = TimmClassifier.load_from_checkpoint(best_checkpoint)
    else:
        log.warning("No checkpoints found! Using initialized model weights.")

    model.eval()

    # Add this section to load class labels
    class_labels = cfg.data.class_names if hasattr(cfg.data, 'class_names') else None
    if class_labels is None:
        log.warning("No class labels found in config. Using index as label.")

    # Set up image transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Get input images
    input_dir = os.path.join(cfg.paths.model_artifacts, "input_images")
    output_dir = os.path.join(cfg.paths.model_artifacts, "predicted_images")
    os.makedirs(output_dir, exist_ok=True)

    image_files = glob.glob(os.path.join(input_dir, "*.jpg"))[:10]  # Get first 10 jpg images

    for img_path in image_files:
        # Load and preprocess the image
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0)

        # Make prediction
        with torch.no_grad():
            output = model(img_tensor)
            _, predicted_idx = torch.max(output, 1)
            if class_labels:
                predicted_label = class_labels[predicted_idx.item()]
            else:
                predicted_label = f"Class {predicted_idx.item()}"

        # Create a figure with the image and prediction
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        plt.title(f"Predicted: {predicted_label}")
        plt.axis('off')

        # Save the figure
        output_path = os.path.join(output_dir, f"pred_{os.path.basename(img_path)}")
        plt.savefig(output_path)
        plt.close()

        log.info(f"Processed {img_path} -> {output_path}")

    log.info("Inference complete")

@hydra.main(version_base="1.3", config_path="../configs", config_name="infer.yaml")
def main(cfg: DictConfig) -> None:
    # Setup logger for the entire script
    logging_utils.setup_logger(log_file=f"{cfg.paths.output_dir}/infer.log")
    
    # Run inference
    infer(cfg)

if __name__ == "__main__":
    main()
