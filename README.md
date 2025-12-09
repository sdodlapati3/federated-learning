# FL Research: Federated Learning with Differential Privacy

[![CI](https://github.com/sdodlapati3/federated-learning/actions/workflows/ci.yml/badge.svg)](https://github.com/sdodlapati3/federated-learning/actions/workflows/ci.yml)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Flower 1.24+](https://img.shields.io/badge/Flower-1.24+-orange.svg)](https://flower.ai/)
[![Opacus 1.5+](https://img.shields.io/badge/Opacus-1.5+-green.svg)](https://opacus.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A comprehensive library and learning resource for **Federated Learning (FL)** with **Differential Privacy (DP)**. Built on top of [Flower](https://flower.ai/), [PyTorch](https://pytorch.org/), and [Opacus](https://opacus.ai/).

## 🎯 Features

- **Reusable FL Components**: Models, data partitioners, strategies, and privacy tools
- **Multiple FL Strategies**: FedAvg, FedProx, SCAFFOLD with variance reduction
- **Differential Privacy**: RDP accounting, noise mechanisms, Opacus integration
- **Non-IID Data Simulation**: Dirichlet, pathological, and shard-based partitioning
- **Experiment Tracking**: Metrics, checkpointing, and reproducibility utilities
- **Comprehensive Documentation**: Learning guides, algorithm notes, framework comparisons
- **105 Unit Tests**: Extensive test coverage with CI/CD pipeline

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/sdodlapati3/federated-learning.git
cd federated-learning

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac

# Install the library in development mode
pip install -e .

# Or install dependencies only
pip install -r requirements.txt
```

## 🚀 Quick Start

### Using the Library

```python
from fl_research.models import ModelRegistry, CIFAR10CNN
from fl_research.data import load_cifar10, DirichletPartitioner
from fl_research.privacy import PrivacyAccountant
from fl_research.strategies import SCAFFOLDServer, StandaloneSCAFFOLDClient
from fl_research.utils import set_seed, get_device, MetricsTracker

# Set up reproducibility
set_seed(42)
device = get_device()

# Load data with non-IID partitioning
train_data, test_data = load_cifar10()
partitioner = DirichletPartitioner(num_clients=10, alpha=0.5)
partitions = partitioner.partition(train_data)

# Create model from registry
model = ModelRegistry.create('cifar10cnn').to(device)

# Track metrics
tracker = MetricsTracker()
tracker.log_round({'round': 1, 'accuracy': 0.85, 'loss': 0.45})
```

### Available Models

```python
from fl_research.models import ModelRegistry

# List all available models
print(ModelRegistry.list())
# ['simplecnn', 'cifar10cnn', 'cifar10cnn_opacus', 'resnet_small', 'mlp', 'twolayer_mlp']

# Create models
cnn = ModelRegistry.create('cifar10cnn')
dp_cnn = ModelRegistry.create('cifar10cnn_opacus')  # DP-compatible (GroupNorm)
mlp = ModelRegistry.create('mlp', input_dim=784, hidden_dims=[256, 128], num_classes=10)
```

### Data Partitioning

```python
from fl_research.data import IIDPartitioner, DirichletPartitioner, ShardPartitioner

# IID partitioning (uniform distribution)
iid = IIDPartitioner(num_clients=10)

# Non-IID with Dirichlet distribution (alpha controls heterogeneity)
dirichlet = DirichletPartitioner(num_clients=10, alpha=0.1)  # Very heterogeneous
dirichlet = DirichletPartitioner(num_clients=10, alpha=1.0)  # Moderate

# Shard-based partitioning
shard = ShardPartitioner(num_clients=10, shards_per_client=2)

# Apply to dataset
partitions = dirichlet.partition(train_dataset)  # Returns list of index lists
```

### Privacy Accounting

```python
from fl_research.privacy import PrivacyAccountant, get_privacy_spent

# Create accountant with budget
accountant = PrivacyAccountant(target_epsilon=10.0, target_delta=1e-5)

# Track privacy consumption
for round in range(100):
    accountant.step(noise_multiplier=1.0, sample_rate=0.01, num_steps=10)
    
    if accountant.is_budget_exceeded():
        print(f"Budget exceeded at round {round}")
        break

# Get current epsilon
epsilon = accountant.get_epsilon()
```

### FL Strategies

```python
from fl_research.strategies import SCAFFOLDServer, StandaloneSCAFFOLDClient

# SCAFFOLD for variance reduction
server = SCAFFOLDServer(model, device)
client = StandaloneSCAFFOLDClient(client_id=0, dataloader=loader, device=device)
client.initialize_control_variate(model)

# Train with gradient correction
delta_w, delta_c, count = client.train(model, server.global_control, epochs=5, lr=0.1)
server.aggregate([delta_w], [delta_c], [count], total_clients=10)
```

## 📁 Project Structure

```
flower-federated-learning/
├── src/fl_research/           # Main library
│   ├── models/                # CNN, MLP, ResNet models
│   │   ├── cnn.py            # SimpleCNN, CIFAR10CNN, CIFAR10CNNOpacus
│   │   ├── mlp.py            # MLP, TwoLayerMLP
│   │   └── registry.py       # ModelRegistry factory
│   ├── data/                  # Data loading and partitioning
│   │   ├── loaders.py        # CIFAR-10, MNIST, Fashion-MNIST
│   │   └── partitioners.py   # IID, Dirichlet, Shard partitioners
│   ├── privacy/               # Differential privacy
│   │   ├── accountant.py     # RDP-based privacy accounting
│   │   ├── mechanisms.py     # Gaussian, Laplace noise
│   │   └── opacus_utils.py   # Opacus integration helpers
│   ├── strategies/            # FL algorithms
│   │   ├── fedavg.py         # Federated Averaging
│   │   ├── fedprox.py        # FedProx with proximal term
│   │   └── scaffold.py       # SCAFFOLD with control variates
│   └── utils/                 # Utilities
│       ├── config.py         # YAML/JSON configuration
│       ├── metrics.py        # MetricsTracker
│       ├── checkpointing.py  # Model checkpoints
│       └── reproducibility.py # Seeds, device detection
├── tests/                     # Test suite (105 tests)
│   ├── test_data.py          # Data loading tests
│   ├── test_models.py        # Model registry tests
│   ├── test_privacy.py       # Privacy accounting tests
│   ├── test_utils.py         # Utility tests
│   ├── test_strategies.py    # FedAvg, FedProx, SCAFFOLD tests
│   └── test_integration.py   # End-to-end workflow tests
├── examples/                  # Example experiments
│   ├── dp/                   # Differential privacy experiments
│   ├── scaffold/             # SCAFFOLD vs FedAvg comparison
│   ├── fedprox/              # FedProx experiments
│   ├── minimal_example.py    # Quick runnable demo
│   └── ...                   # Additional examples
├── docs/                      # 📚 Documentation
│   ├── FEDERATED_LEARNING_GUIDE.md  # Comprehensive FL learning guide
│   ├── DP_VARIANTS_NOTES.md         # DP variants explained
│   ├── DISTRIBUTED_OPTIMIZATION_NOTES.md  # Optimization theory
│   ├── HPC_SCALING_REPORT.md        # HPC scaling patterns
│   ├── FEDML_COMPARISON.md          # Framework comparison
│   └── COMPLETION_REPORT.md         # Project summary
├── .github/workflows/ci.yml   # CI/CD pipeline
├── README.md
├── pyproject.toml
└── requirements.txt
```

## 📊 Experiments

### Differential Privacy Experiments

```bash
cd examples/dp
python run_standalone_experiments_refactored.py
```

Compares IID vs Non-IID data with varying privacy levels (ε = 2, 4, 8).

### SCAFFOLD vs FedAvg

```bash
cd examples/scaffold
python scaffold_implementation_refactored.py
```

Demonstrates variance reduction with control variates on heterogeneous data.

### FedProx Comparison

```bash
cd examples/fedprox
python fedprox_implementation_refactored.py
```

Shows proximal term benefits for client drift mitigation.

## 🧪 Testing

```bash
# Run all tests (105 tests)
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=fl_research --cov-report=html

# Run specific test files
pytest tests/test_strategies.py -v  # Strategy tests
pytest tests/test_integration.py -v  # End-to-end tests
```

## 📚 Learning Path

This repository follows a structured learning journey:

| Week | Topic | Key Concepts |
|------|-------|--------------|
| 1-2 | Flower Basics | Client/Server, FedAvg, Simulation |
| 3 | Differential Privacy | ε-δ DP, RDP, Opacus integration |
| 4 | DP + FL | Privacy-utility tradeoffs, composition |
| 5-6 | Advanced Strategies | FedProx, SCAFFOLD, Non-IID handling |

## 🔧 Configuration

```python
from fl_research.utils import Config, save_config, load_config

# Create experiment config
config = Config(
    num_rounds=100,
    num_clients=10,
    batch_size=32,
    learning_rate=0.01,
    strategy='scaffold',
    privacy={'epsilon': 10.0, 'delta': 1e-5}
)

# Save and load
save_config(config, 'experiment.yaml')
loaded = load_config('experiment.yaml')
```

## 📚 Documentation

See the [`docs/`](docs/) folder for comprehensive learning resources:

| Document | Description |
|----------|-------------|
| [FEDERATED_LEARNING_GUIDE.md](docs/FEDERATED_LEARNING_GUIDE.md) | 📖 Complete FL+DP learning guide with citations |
| [DP_VARIANTS_NOTES.md](docs/DP_VARIANTS_NOTES.md) | Approximate, Rényi, Local DP explained |
| [DISTRIBUTED_OPTIMIZATION_NOTES.md](docs/DISTRIBUTED_OPTIMIZATION_NOTES.md) | Optimization theory for FL |
| [HPC_SCALING_REPORT.md](docs/HPC_SCALING_REPORT.md) | HPC scaling patterns |
| [FEDML_COMPARISON.md](docs/FEDML_COMPARISON.md) | Framework comparison guide |

## 📖 References

### Papers
- [FedAvg](https://arxiv.org/abs/1602.05629) - Communication-Efficient Learning
- [FedProx](https://arxiv.org/abs/1812.06127) - Heterogeneous Federated Optimization
- [SCAFFOLD](https://arxiv.org/abs/1910.06378) - Stochastic Controlled Averaging
- [DP-SGD](https://arxiv.org/abs/1607.00133) - Deep Learning with Differential Privacy
- [Awesome Federated Learning](https://github.com/FedML-AI/FedML/blob/master/research/Awesome-Federated-Learning.md) - 500+ papers

### Documentation
- [Flower Framework](https://flower.ai/docs/framework/)
- [Opacus Library](https://opacus.ai/)
- [PyTorch](https://pytorch.org/docs/)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## 📝 License

MIT License - See [LICENSE](LICENSE) for details.

---

**Built with ❤️ for the Federated Learning research community**
