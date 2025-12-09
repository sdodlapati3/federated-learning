# Flower Federated Learning Tutorial Plan

## Overview

This repository follows the [Flower PyTorch Tutorial](https://flower.ai/docs/framework/tutorial-series-get-started-with-flower-pytorch.html) to learn federated learning concepts and implementation.

**Goal**: Build a federated learning system using Flower framework with PyTorch, training a CNN on CIFAR-10 dataset across 10 simulated clients.

---

## Phase 1: Environment Setup

### 1.1 Create Python Virtual Environment

```bash
# Create a dedicated environment for Flower
python -m venv flower-env
source flower-env/bin/activate  # Linux/Mac
# OR
flower-env\Scripts\activate     # Windows
```

### 1.2 Install Dependencies

```bash
# Install Flower with simulation support
pip install -U "flwr[simulation]"

# Verify installation
pip show flwr
```

### 1.3 Create Flower Project from Template

```bash
# Create new Flower app using PyTorch template
flwr new flower-tutorial --framework pytorch --username flwrlabs

# Navigate and install project dependencies
cd flower-tutorial
pip install -e .
```

---

## Phase 2: Understanding the Project Structure

After running `flwr new`, the project structure will be:

```
flower-tutorial/
├── README.md
├── flower_tutorial/
│   ├── __init__.py
│   ├── client_app.py   # Defines ClientApp (client-side logic)
│   ├── server_app.py   # Defines ServerApp (server-side logic)
│   └── task.py         # Model, training, and data loading
├── pyproject.toml      # Project metadata and configs
└── README.md
```

### Key Components:

| File | Purpose |
|------|--------|
| `task.py` | CNN model definition, train/test functions, data loading |
| `client_app.py` | ClientApp with `@app.train()` and `@app.evaluate()` decorators |
| `server_app.py` | ServerApp with FedAvg strategy configuration |
| `pyproject.toml` | Dependencies, run configs, federation settings |

---

## Phase 3: Tutorial Implementation Steps

### Step 1: Understand the CIFAR-10 Dataset
- Dataset: 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- Federated setup: Split into 10 partitions (one per simulated client)
- Each client gets: 4000 training + 1000 validation examples

### Step 2: Review the CNN Model (`task.py`)
```python
class Net(nn.Module):
    """Simple CNN from PyTorch 60-minute Blitz"""
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
```

### Step 3: Review Training and Test Functions
- `train()`: Standard PyTorch training loop with CrossEntropyLoss + Adam
- `test()`: Evaluation loop returning loss and accuracy

### Step 4: Understand Flower Message System
- **ArrayRecord**: Model parameters (NumPy arrays)
- **MetricRecord**: Training/evaluation metrics (int/float/lists)
- **ConfigRecord**: Configuration parameters (int/float/str/bool)
- **RecordDict**: Container for records (main Message payload)

### Step 5: Implement ClientApp (`client_app.py`)

#### Training Function (`@app.train()`)
1. Receive Message with global model parameters
2. Load local data partition
3. Train model locally
4. Return updated parameters + metrics

#### Evaluation Function (`@app.evaluate()`)
1. Receive Message with global model parameters
2. Load local validation data
3. Evaluate model
4. Return metrics (loss, accuracy, num-examples)

### Step 6: Implement ServerApp (`server_app.py`)
1. Initialize global model
2. Configure FedAvg strategy
3. Run federated training rounds
4. Save final model

### Step 7: Run the Simulation

```bash
# Basic run (uses pyproject.toml defaults)
flwr run .

# Custom configuration
flwr run . --run-config "num-server-rounds=5 local-epochs=3"
```

---

## Phase 4: Expected Output

After running `flwr run .`, you should see:

```
INFO :      Starting FedAvg strategy:
INFO :          ├── Number of rounds: 3
INFO :          ├── ArrayRecord (0.24 MB)
...
INFO :      [ROUND 1/3]
INFO :      configure_train: Sampled 5 nodes (out of 10)
INFO :      aggregate_train: Received 5 results and 0 failures
INFO :          └──> Aggregated MetricRecord: {'train_loss': 2.25811}
...
INFO :      [ROUND 3/3]
...
INFO :      Strategy execution finished in 17.18s

Saving final model to disk...
```

---

## Phase 5: Next Steps (Advanced Tutorials)

After completing Part 1, continue with:

1. **[Part 2: Federated Learning Strategies](https://flower.ai/docs/framework/tutorial-series-use-a-federated-learning-strategy-pytorch.html)**
   - Customize FedAvg parameters
   - Learning rate decay from server
   - Server-side model evaluation

2. **[Part 3: Build a Strategy from Scratch](https://flower.ai/docs/framework/tutorial-series-build-a-strategy-from-scratch-pytorch.html)**
   - Custom aggregation logic
   - Implement new FL algorithms

3. **[Part 4: Custom Messages](https://flower.ai/docs/framework/tutorial-series-customize-the-client-pytorch.html)**
   - Custom client-server communication
   - Advanced message handling

---

## Checklist

- [ ] Create Python virtual environment
- [ ] Install Flower with simulation support
- [ ] Create project from PyTorch template
- [ ] Review and understand `task.py`
- [ ] Review and understand `client_app.py`
- [ ] Review and understand `server_app.py`
- [ ] Run basic simulation (`flwr run .`)
- [ ] Experiment with different configurations
- [ ] Complete advanced tutorials (Parts 2-4)

---

## Resources

- [Flower Documentation](https://flower.ai/docs/framework/)
- [Flower GitHub](https://github.com/adap/flower)
- [Flower Discuss Forum](https://discuss.flower.ai/)
- [Flower Slack](https://flower.ai/join-slack)
- [PyTorch 60-Minute Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
