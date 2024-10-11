# Dog Breed Classifier using PyTorch Lightning, Hydra and Docker

This project is a dog breed classifier that uses PyTorch Lightning, Hydra and Docker. It is a simple example of how to use these technologies to build a machine learning model. It is not meant to be a production-ready application.

It is an adaptation of the [my earlier repository](https://github.com/mkthoma/pytorch_lightning_docker) to showcase the use of Hydra for configuration management and Docker for containerization.

Before you start, make sure you have the following:


1. ```
    touch .project-root
    ```

2.  ```
    export UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu
    ```
3.  ```
    uv sync --index-strategy unsafe-best-match
    ```
4.  ```
    echo 'export KAGGLE_USERNAME="your_username"' >> ~/.bashrc
    echo 'export KAGGLE_KEY="your_api_key"' >> ~/.bashrc
    source ~/.bashrc
    ```

## Docker commands
Based on the provided Dockerfile, the Docker commands for building the image and running train, eval, and infer with volume mounts for model artifacts. Here are the commands:

1. Build the Docker image:

    ```
    docker build -t dogbreed-classifier .
    ```

2. Run the train command:

    ```
    docker run --rm -v $(pwd)/logs:/app/logs -v $(pwd)/data:/app/data dogbreed-classifier src/train.py
    ```

3.  Run the eval command:

    ```
    docker run --rm -v $(pwd)/logs:/app/logs -v $(pwd)/data:/app/data dogbreed-classifier src/eval.py
    ``` 

4. Run the infer command:

    ```
    docker run --rm -v $(pwd)/model_artifacts:/app/model_artifacts -v $(pwd)/logs:/app/logs -v $(pwd)/data:/app/data dogbreed-classifier src/infer.py 
    ```

## Testing

This project uses pytest for unit testing and code coverage analysis. The tests are located in the `tests` folder.

### Running Tests

To run the tests, use the following command:
```
pytest tests/
```
To run individual tests, you can use the following command:
```
pytest tests/test_train.py
```

```
pytest tests/test_eval.py
```

```
pytest tests/test_infer.py
```

```
pytest tests/test_models/test_timm_classifier.py
```

```
pytest tests/test_datamodules/test_dogbreed.py
```

```
pytest tests/test_models/test_timm_classifier.py
```
## Code Coverage

To generate the code coverage report, use the following command:

```
coverage run -m pytest
coverage report -m
```

![image](https://github.com/user-attachments/assets/9bcea1c0-d25c-46a8-86bb-820cca1bee3f)


The CodeCov report can be found [here](https://app.codecov.io/github/mkthoma/lightning_hydra).


## Continuous Integration

This project uses GitHub Actions for continuous integration. The workflow is defined in [`.github/workflows/ci.yml`](https://github.com/mkthoma/lightning_hydra/blob/main/.github/workflows/ci.yml). Here's an overview of the CI/CD pipeline:

### Workflow Triggers
The pipeline is triggered on pushes to the `main` branch, specifically when changes are made to:
- `.github/workflows/ci.yml`
- `Dockerfile`
- Any file in the `src/` directory

### Jobs

1. **Test**
   - Runs on Ubuntu latest
   - Sets up Python 3.12 with uv
   - Installs dependencies
   - Sets up Kaggle credentials (securely using GitHub Secrets)
   - Runs tests with pytest and generates coverage report
   - Uploads coverage to Codecov

2. **Build and Push Image**
   - Runs after the test job completes successfully
   - Logs in to the GitHub Container Registry
   - Builds the Docker image
   - Pushes the image to the GitHub Container Registry

### Environment Variables
- `REGISTRY`: Set to `ghcr.io` (GitHub Container Registry)
- `IMAGE_NAME`: Set to the GitHub repository name

### Secrets Used
- `KAGGLE_USERNAME`: Your Kaggle username (set in GitHub Secrets)
- `KAGGLE_KEY`: Your Kaggle API key (set in GitHub Secrets)
- `CODECOV_TOKEN`: Token for uploading coverage to Codecov (set in GitHub Secrets)

This CI/CD pipeline ensures that tests are run on every relevant push to the main branch, and if tests pass, a new Docker image is built and pushed to the GitHub Container Registry.


## Model Artifacts Storage

This project uses GitHub Actions to train the dog breed classifier model and store various artifacts. The workflow is defined in [`.github/workflows/train_dogbreed_model.yml`](https://github.com/mkthoma/lightning_hydra/blob/main/.github/workflows/train_dogbreed_model.yml). Here's how different model artifacts are stored:

1. **Model Checkpoint**
   - The trained model checkpoint is uploaded as an artifact.
   - Artifact name: `model-checkpoint`
   - Path: `logs/train/runs/*/checkpoints/*.ckpt`

2. **Training Logs**
   - The training logs are uploaded as an artifact.
   - Artifact name: `model-logs`
   - Path: `logs/train/runs/*/train.log`

3. **Model Configuration**
   - The model's hyperparameters and configuration are uploaded.
   - Artifact name: `model-config`
   - Path: `logs/train/runs/*/csv/version_0/hparams.yaml`

4. **TensorBoard Logs**
   - TensorBoard logs for visualizing training progress are uploaded.
   - Artifact name: `tensorboard-logs`
   - Path: `logs/train/runs/*/tensorboard`

5. **Test Results**
   - The model's test results are uploaded as an artifact.
   - Artifact name: `test-results`
   - Path: `logs/train/runs/*/test_results.pt`

These artifacts are automatically uploaded and stored in GitHub Actions after each successful training run. They can be accessed and downloaded from the GitHub Actions workflow run page, allowing for easy tracking of model versions, configurations, and performance across different runs.

To access these artifacts:
1. Go to the "Actions" tab in your GitHub repository.
2. Click on the specific workflow run.
3. Scroll down to the "Artifacts" section.
4. Download the desired artifact.

This approach ensures that all important model-related files are preserved and easily accessible for further analysis or deployment.
