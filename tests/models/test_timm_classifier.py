import pytest
import torch
import rootutils
# Setup root directory
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
print(f"Project root: {root}")

from src.models.timm_classifier import TimmClassifier

def test_timm_classifier():
    model = TimmClassifier(model_name="resnet18", num_classes=10)
    assert isinstance(model, TimmClassifier)
    
    # Test forward pass
    x = torch.randn(1, 3, 224, 224)
    output = model(x)
    assert output.shape == (1, 10)
