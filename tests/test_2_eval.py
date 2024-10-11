import os
import pytest
from hydra import initialize, compose
import rootutils
from typing import List

# Setup root directory
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
print(f"Project root: {root}")

from src.eval import evaluate

def test_evaluate():
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="eval.yaml")
    
    # Override paths in the config
    project_root = rootutils.find_root(indicator=".project-root")
    cfg.paths.root_dir = str(project_root)
    cfg.paths.data_dir = str(project_root / "data")
    cfg.paths.log_dir = str(project_root / "logs")
    cfg.paths.output_dir = os.path.join(cfg.paths.log_dir, "pytest")
    cfg.paths.work_dir = str(project_root)

    # Create the output directory if it doesn't exist
    os.makedirs(cfg.paths.output_dir, exist_ok=True)

    # Modify config for fast dev run
    cfg.trainer.fast_dev_run = True

    print("Eval Model Config: ", cfg) 
    # Run evaluation
    try:
        eval_metrics = evaluate(cfg)
        print("Eval_metrics: ", eval_metrics)
        
        assert isinstance(eval_metrics, dict), "Evaluation should return a dictionary"
        assert "validation_metrics" in eval_metrics, "Validation metrics should be in the results"
        assert "callback_metrics" in eval_metrics, "Callback metrics should be in the results"

        validation_metrics = eval_metrics["validation_metrics"]
        callback_metrics = eval_metrics["callback_metrics"]

        assert "val/loss" in validation_metrics, "Validation loss should be in metrics"
        assert "val/acc" in validation_metrics, "Validation accuracy should be in metrics"

        # Check if metric values are within expected ranges
        assert 0 <= validation_metrics["val/loss"] <= 10, "Validation loss should be between 0 and 10"
        assert 0 <= validation_metrics["val/acc"] <= 1, "Validation accuracy should be between 0 and 1"

        # Check if evaluation results are saved
        results_file = os.path.join(cfg.paths.output_dir, "test_results.pt")
        assert os.path.exists(results_file), "Evaluation results file should exist"
        
        # Optional: Add checks for callback metrics if needed
        # For example:
        # assert "some_callback_metric" in callback_metrics, "Expected callback metric not found"

    except Exception as e:
        pytest.fail(f"Evaluation failed with error: {str(e)}")
