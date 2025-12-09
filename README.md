# Flower Federated Learning Tutorial

[![Flower](https://img.shields.io/badge/Flower-Framework-orange)](https://flower.ai/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue)](https://python.org/)

Learn federated learning with the [Flower](https://flower.ai/) framework using PyTorch.

## 📚 Tutorial Link

**Official Tutorial**: [Get Started with Flower (PyTorch)](https://flower.ai/docs/framework/tutorial-series-get-started-with-flower-pytorch.html)

### Tutorial Series

| Part | Topic | Link |
|------|-------|------|
| 1 | Get Started with Flower | [Tutorial](https://flower.ai/docs/framework/tutorial-series-get-started-with-flower-pytorch.html) |
| 2 | Use a Federated Learning Strategy | [Tutorial](https://flower.ai/docs/framework/tutorial-series-use-a-federated-learning-strategy-pytorch.html) |
| 3 | Build a Strategy from Scratch | [Tutorial](https://flower.ai/docs/framework/tutorial-series-build-a-strategy-from-scratch-pytorch.html) |
| 4 | Customize the Client | [Tutorial](https://flower.ai/docs/framework/tutorial-series-customize-the-client-pytorch.html) |

---

## 🚀 Quick Start

### Option 1: Using Setup Script

```bash
# Clone the repository
git clone https://github.com/sdodlapati3/flower-federated-learning.git
cd flower-federated-learning

# Run setup script
chmod +x setup.sh
./setup.sh
```

### Option 2: Manual Setup

```bash
# 1. Create and activate virtual environment
python -m venv flower-env
source flower-env/bin/activate  # Linux/Mac
# flower-env\Scripts\activate   # Windows

# 2. Install dependencies
pip install -U "flwr[simulation]" torch torchvision

# 3. Create Flower project from template
flwr new flower-tutorial --framework pytorch --username your-username

# 4. Install and run
cd flower-tutorial
pip install -e .
flwr run .
```

---

## 📁 Project Structure

After running `flwr new`, you'll have:

```
flower-tutorial/
├── flower_tutorial/
│   ├── __init__.py
│   ├── client_app.py   # Client-side federated logic
│   ├── server_app.py   # Server-side aggregation logic
│   └── task.py         # Model, training, data loading
├── pyproject.toml      # Project config and dependencies
└── README.md
```

---

## 🎯 What You'll Learn

1. **Federated Learning Basics**
   - Split data across multiple clients
   - Train local models on each client
   - Aggregate models on a central server

2. **Flower Framework Components**
   - `ClientApp`: Local training and evaluation
   - `ServerApp`: Coordination and aggregation
   - `FedAvg`: Federated averaging strategy

3. **Key Concepts**
   - `ArrayRecord`: Model parameters
   - `MetricRecord`: Training/eval metrics
   - `ConfigRecord`: Configuration settings
   - `Message`: Client-server communication

---

## 🔧 Running Experiments

```bash
# Activate environment
source flower-env/bin/activate
cd flower-tutorial

# Basic run (3 rounds, 10 clients)
flwr run .

# Custom configuration
flwr run . --run-config "num-server-rounds=5 local-epochs=3"

# More rounds with higher participation
flwr run . --run-config "num-server-rounds=10 fraction-train=0.8"
```

---

## 📊 Expected Output

```
INFO :      Starting FedAvg strategy:
INFO :          ├── Number of rounds: 3
INFO :          ├── ArrayRecord (0.24 MB)
...
INFO :      [ROUND 1/3]
INFO :      configure_train: Sampled 5 nodes (out of 10)
INFO :      aggregate_train: Received 5 results and 0 failures
INFO :          └──> Aggregated MetricRecord: {'train_loss': 2.25811}
INFO :      configure_evaluate: Sampled 10 nodes (out of 10)
INFO :          └──> Aggregated MetricRecord: {'eval_loss': 2.30, 'eval_acc': 0.10}
...
INFO :      [ROUND 3/3]
...
INFO :      Strategy execution finished in 17.18s

Saving final model to disk...
```

---

## 📖 Documentation

- [PLAN.md](PLAN.md) - Detailed implementation plan
- [Flower Docs](https://flower.ai/docs/framework/)
- [Flower GitHub](https://github.com/adap/flower)
- [Flower Discuss](https://discuss.flower.ai/)
- [Flower Slack](https://flower.ai/join-slack)

---

## 🛠️ Requirements

- Python 3.8+
- PyTorch 2.0+
- Flower 1.24+

See [requirements.txt](requirements.txt) for full dependencies.

---

## 📝 License

MIT License - See [LICENSE](LICENSE) for details.
