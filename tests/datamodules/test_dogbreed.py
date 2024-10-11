import pytest
from hydra import initialize, compose
import rootutils
# Setup root directory
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
print(f"Project root: {root}")

from src.datamodules.dogbreed import DogBreedImageDataModule

@pytest.fixture(scope="module")
def hydra_config():
    with initialize(version_base="1.3", config_path="../../configs"):
        cfg = compose(config_name="train.yaml")
        return cfg

def test_dogbreed_datamodule(hydra_config):
    data_config = hydra_config.data
    # Remove Hydra-specific keys
    data_config_dict = {k: v for k, v in data_config.items() if not k.startswith('_')}
    
    datamodule = DogBreedImageDataModule(**data_config_dict)
    
    # Prepare data and setup
    datamodule.prepare_data()
    datamodule.setup()
    
    # Check dataset lengths
    assert len(datamodule.train_dataset) > 0
    assert len(datamodule.val_dataset) > 0
    assert len(datamodule.test_dataset) > 0
    
    # Check dataloader functionality
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()
    
    assert len(train_loader) > 0
    assert len(val_loader) > 0
    assert len(test_loader) > 0
