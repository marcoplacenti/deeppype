# DeepPype

An end-to-end deep learning pipeline framework built on AWS, designed to automate the full lifecycle of image classification workflows — from data ingestion to model deployment.

Developed as a special project at DTU (Technical University of Denmark) in Spring 2022, in collaboration with Novo Nordisk, Data Management & Analytics.

---

## Overview

DeepPype orchestrates deep learning pipelines through a single, configuration-driven interface. It handles:

- **Data preprocessing** — load from local filesystem or S3, resize, split into train/test sets
- **Model training** — PyTorch Lightning-based training with early stopping and checkpointing
- **Hyperparameter tuning** — automated search via Ray Tune with ASHA scheduling
- **Evaluation** — test set accuracy reporting
- **Deployment** — package and serve models on AWS SageMaker endpoints
- **Experiment tracking** — automatic logging to Weights & Biases

---

## Installation

```bash
pip install deeppype
```

**Requirements:** Python 3.9+

---

## Prerequisites

DeepPype requires credentials for AWS (IAM, S3, SageMaker) and Weights & Biases. Set the following environment variables before running:

```bash
export CLOUD_DEVOPS_USER_ACCESS_KEY_ID=<your-aws-access-key>
export CLOUD_DEVOPS_USER_SECRET_ACCESS_KEY=<your-aws-secret-key>
export WANDB_KEY=<your-wandb-api-key>
```

You can also place these in a `.env` file in your working directory.

---

## Usage

### Option 1: Configuration file (recommended)

Define your pipeline in a YAML file and launch it in one call:

```python
from deeppype import Pipe

Pipe().launch(config_file='config.yaml')
```

### Option 2: Programmatic API

Build the pipeline step by step:

```python
from deeppype import Pipe

pipe = Pipe()
pipe.launch(name='mnist-experiment', task='Classification')

pipe.preproc_data(
    location='./data/mnist/',
    img_res=[28, 28],
    greyscale=True,
    test_size=0.2,
    folds=0
)

pipe.train(
    model='VGG',
    max_epochs=20,
    batch_size=32,
    optimizer='adam',
    learning_rate=0.001
)

pipe.eval()

pipe.deploy(
    tarfile_name='./tmp/myModel.tar.gz',
    endpoint_name='mnist-endpoint',
    instance_type='ml.m4.xlarge'
)
```

### Option 3: Run predictions

```python
from deeppype import Pipe

results = Pipe().predict(
    project_name='mnist-experiment',
    endpoint='mnist-endpoint',
    images_path=['./samples/img1.jpg', './samples/img2.jpg']
)
```

---

## Configuration File Reference

```yaml
project:
  name: my-project           # Project name (used for W&B and S3 paths)
  task: Classification       # Task type (only Classification is supported)
  experiment: run-01         # Optional — auto-generated if omitted

data:
  location: ./data/mnist/    # Local path or S3:// URI
  img-res: [28, 28]          # Target image resolution [height, width]
  greyscale: true            # true for grayscale, false for RGB

model_architecture:
  name: VGG                  # SimpleModel or VGG

training:
  max_epochs: 20
  batch_size: 32             # Single value or list [16, 32, 64] for tuning

optimization:
  optimizer: adam            # adam or sgd; use a list for tuning
  learning_rate: 0.001       # Single value or [min, max] range for tuning
  loss_fnc: NLLLoss

tuning:
  number_trials: 10          # Number of Ray Tune trials (ignored if no HP ranges set)

validation:
  test_size: 0.2             # Fraction of data reserved for testing
  folds: 5                   # K-fold splits; set to null to disable cross-validation

deployment:
  min_test_score: 0.85       # Minimum test accuracy required to trigger deployment
  endpoint_name: my-endpoint
  instance_type: ml.m4.xlarge
```

### Enabling hyperparameter tuning

Pass lists (or ranges) for any of `batch_size`, `optimizer`, or `learning_rate` to activate Ray Tune:

```yaml
training:
  batch_size: [16, 32, 64]

optimization:
  optimizer: [adam, sgd]
  learning_rate: [0.0001, 0.01]   # Sampled log-uniformly
```

---

## Data Format

DeepPype expects image data organized by class in subdirectories. Two layouts are supported:

**Pre-split (training/testing directories):**
```
data/
├── training/
│   ├── cat/
│   └── dog/
└── testing/
    ├── cat/
    └── dog/
```

**Unsplit (split handled automatically):**
```
data/
├── cat/
└── dog/
```

---

## Available Architectures

| Name | Description |
|---|---|
| `SimpleModel` | Lightweight 2-layer CNN — fast training, suitable for simple tasks |
| `VGG` | 4-block VGG-style CNN with batch normalization — higher capacity |

---

## Pipeline Architecture

```
launch(config_file)
    │
    ├─ Configurator       →  Validates YAML configuration
    ├─ AWSConnector       →  Sets up IAM / S3 / SageMaker session
    │
    ├─ preproc_data()     →  Loads + resizes images, creates train/test splits
    │                        Uploads processed data to S3
    │
    ├─ train()
    │   ├─ [HP ranges]    →  Ray Tune ASHA search across trials
    │   └─ train_opt()    →  Final training run with best config
    │                        Saves TorchScript model, uploads checkpoints to S3
    │
    ├─ eval()             →  Tests model, returns accuracy
    │
    └─ deploy()           →  Packages model + inference code into tar.gz
                             Uploads to SageMaker, creates serving endpoint
```

---

## Tech Stack

| Component | Library |
|---|---|
| Model training | [PyTorch](https://pytorch.org/) + [PyTorch Lightning](https://lightning.ai/) |
| Hyperparameter tuning | [Ray Tune](https://docs.ray.io/en/latest/tune/index.html) |
| Cloud infrastructure | [AWS S3](https://aws.amazon.com/s3/) + [SageMaker](https://aws.amazon.com/sagemaker/) |
| Experiment tracking | [Weights & Biases](https://wandb.ai/) |
| Configuration | YAML |

---

## License

MIT
