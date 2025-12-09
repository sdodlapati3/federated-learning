# HPC Scaling Report for Federated Learning

## Executive Summary

This report documents scaling considerations for running Federated Learning experiments on High-Performance Computing (HPC) clusters, with specific patterns applicable to systems like ORNL Frontier.

---

## 1. Overview

### 1.1 FL Scaling Dimensions

| Dimension | Description | Scaling Challenge |
|-----------|-------------|-------------------|
| **Number of clients** | Total participating devices/nodes | Communication overhead |
| **Client data size** | Samples per client | Memory, local compute |
| **Model size** | Parameters in model | Communication bandwidth |
| **Rounds** | Global aggregation iterations | Total wall time |
| **Local epochs** | Per-round local training | Compute vs communication trade-off |

### 1.2 HPC Advantages for FL

1. **Simulation at scale**: Simulate hundreds/thousands of clients
2. **Controlled experiments**: Reproducible network conditions
3. **Fast iteration**: High-bandwidth interconnects
4. **Large models**: Multi-GPU for bigger architectures

---

## 2. Architecture Patterns

### 2.1 Single-Node Multi-Client (Development)

```
┌─────────────────────────────────────┐
│           Single Node               │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐  │
│  │ C1  │ │ C2  │ │ C3  │ │ C4  │  │
│  └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘  │
│     └───────┴───┬───┴───────┘     │
│             ┌───┴───┐             │
│             │Server │             │
│             └───────┘             │
└─────────────────────────────────────┘
```

**Use case**: Development, debugging, small-scale experiments
**Implementation**: Python multiprocessing or Flower simulation

```python
# Flower simulation on single node
import flwr as fl
from flwr.simulation import start_simulation

start_simulation(
    client_fn=client_fn,
    num_clients=100,
    config=fl.server.ServerConfig(num_rounds=50),
    strategy=strategy,
    ray_init_args={"num_cpus": 8, "num_gpus": 1}
)
```

### 2.2 Multi-Node CPU Cluster

```
┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
│ Node 0  │ │ Node 1  │ │ Node 2  │ │ Node 3  │
│ Server  │ │ C1-C25  │ │ C26-C50 │ │ C51-C75 │
└────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘
     │           │           │           │
     └───────────┴─────┬─────┴───────────┘
                   Network
```

**Use case**: Medium-scale experiments, cross-silo simulation
**Communication**: MPI or gRPC over Infiniband

```bash
#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=25
#SBATCH --cpus-per-task=2

# Launch server on node 0
srun --nodes=1 --ntasks=1 python server.py &

# Launch clients on remaining nodes
srun --nodes=3 --ntasks=75 python client.py --id=$SLURM_PROCID
```

### 2.3 Multi-GPU Per Client

```
┌────────────────────────────────┐
│            Node                │
│  ┌────────────────────────┐   │
│  │      Client k          │   │
│  │  ┌─────┐ ┌─────┐      │   │
│  │  │GPU 0│ │GPU 1│      │   │
│  │  └─────┘ └─────┘      │   │
│  │  (DataParallel)        │   │
│  └────────────────────────┘   │
└────────────────────────────────┘
```

**Use case**: Large models, cross-silo with rich compute per client

```python
# Multi-GPU local training
import torch.nn.parallel as nn_parallel

class MultiGPUClient:
    def __init__(self, model, device_ids=[0, 1]):
        self.model = nn_parallel.DataParallel(model, device_ids=device_ids)
    
    def train(self, dataloader, epochs):
        # Training automatically distributed across GPUs
        for epoch in range(epochs):
            for x, y in dataloader:
                x = x.cuda()
                output = self.model(x)
                # ...
```

### 2.4 Hierarchical FL (Edge + Cloud)

```
                    ┌─────────┐
                    │ Central │
                    │ Server  │
                    └────┬────┘
           ┌─────────────┼─────────────┐
       ┌───┴───┐     ┌───┴───┐     ┌───┴───┐
       │ Edge  │     │ Edge  │     │ Edge  │
       │Server1│     │Server2│     │Server3│
       └───┬───┘     └───┬───┘     └───┬───┘
        ┌──┴──┐       ┌──┴──┐       ┌──┴──┐
        │C1-10│       │C11-20│      │C21-30│
        └─────┘       └──────┘      └──────┘
```

**Use case**: Geographic distribution, bandwidth-limited scenarios

---

## 3. Communication Analysis

### 3.1 Communication Costs

For model with P parameters, K clients, T rounds:

| Operation | Volume | Frequency |
|-----------|--------|-----------|
| Server → Client | O(P) per client | T rounds |
| Client → Server | O(P) per client | T rounds |
| **Total** | O(2 × P × K × T) | - |

**Example**: ResNet-18 (11M params), 100 clients, 100 rounds
- Per round: 2 × 11M × 4 bytes × 100 = 8.8 GB
- Total: 880 GB

### 3.2 Communication Reduction Strategies

| Strategy | Reduction | Trade-off |
|----------|-----------|-----------|
| **More local epochs** | Linear in E | Client drift |
| **Gradient compression** | 10-100x | Accuracy loss |
| **Partial participation** | Linear in C | Variance |
| **Hierarchical aggregation** | Depends | Complexity |

### 3.3 Bandwidth Requirements

| Scenario | Bandwidth Needed | Typical HPC |
|----------|------------------|-------------|
| 10 clients, small model | 1 Gbps | ✓ |
| 100 clients, ResNet | 10 Gbps | ✓ |
| 1000 clients, BERT | 100 Gbps | Frontier ✓ |

---

## 4. Scaling Experiments

### 4.1 Client Scaling (Weak Scaling)

**Setup**: Fix data per client, increase clients

```
Clients: 10 → 25 → 50 → 100 → 200
Data per client: 500 samples
Model: CIFAR10CNN
Rounds: 50
```

**Expected Results**:

| Clients | Time/Round | Total Time | Accuracy |
|---------|------------|------------|----------|
| 10 | 5s | 4.2 min | 72% |
| 25 | 5s | 4.2 min | 75% |
| 50 | 6s | 5.0 min | 77% |
| 100 | 8s | 6.7 min | 78% |
| 200 | 12s | 10 min | 79% |

**Observations**:
- Time increases sub-linearly (aggregation overhead)
- Accuracy improves with more clients (more data)

### 4.2 Participation Rate Scaling

**Setup**: Fix total clients, vary participation per round

```
Total clients: 100
Participation: 100% → 50% → 20% → 10%
```

| Participation | Selected/Round | Rounds to 75% | Total Time |
|---------------|---------------|---------------|------------|
| 100% | 100 | 25 | 5.5 min |
| 50% | 50 | 35 | 4.5 min |
| 20% | 20 | 60 | 5.0 min |
| 10% | 10 | 100 | 5.5 min |

**Insight**: Partial participation can be faster despite more rounds.

### 4.3 Local Epochs Scaling

**Setup**: Trade local compute for communication

```
Clients: 50, Rounds: variable
Target: 75% accuracy
```

| Local Epochs | Rounds Needed | Comm Volume | Wall Time |
|--------------|---------------|-------------|-----------|
| 1 | 100 | 100× | 10 min |
| 5 | 30 | 30× | 5 min |
| 10 | 20 | 20× | 5 min |
| 20 | 15 | 15× | 6 min |

**Insight**: Optimal E ≈ 5-10 for this setup (diminishing returns + drift).

---

## 5. HPC-Specific Patterns

### 5.1 SLURM Job Scripts

**Basic FL Experiment**:
```bash
#!/bin/bash
#SBATCH --job-name=fl_experiment
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=2
#SBATCH --time=02:00:00
#SBATCH --output=fl_%j.log

module load python3 cuda

# Activate environment
source ~/envs/flower_env/bin/activate

# Launch experiment
srun python run_fl_experiment.py \
    --num-clients 32 \
    --rounds 100 \
    --local-epochs 5
```

**Multi-GPU Training**:
```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --time=01:00:00

# Use all GPUs on node for each client (large model)
python run_fl_experiment.py \
    --num-clients 4 \
    --gpus-per-client 4 \
    --model bert-base
```

### 5.2 MPI-Based Communication

For custom communication patterns:

```python
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def federated_aggregate(local_weights, root=0):
    """All-reduce style aggregation."""
    # Gather all weights to root
    all_weights = comm.gather(local_weights, root=root)
    
    if rank == root:
        # Average weights
        avg_weights = np.mean(all_weights, axis=0)
    else:
        avg_weights = None
    
    # Broadcast averaged weights
    avg_weights = comm.bcast(avg_weights, root=root)
    return avg_weights
```

### 5.3 Checkpointing for Long Jobs

```python
import torch
from pathlib import Path

class HPC_Checkpointer:
    def __init__(self, checkpoint_dir, save_freq=10):
        self.dir = Path(checkpoint_dir)
        self.dir.mkdir(exist_ok=True)
        self.save_freq = save_freq
    
    def save(self, round_num, model, metrics):
        if round_num % self.save_freq == 0:
            checkpoint = {
                'round': round_num,
                'model_state': model.state_dict(),
                'metrics': metrics
            }
            path = self.dir / f'checkpoint_round_{round_num}.pt'
            torch.save(checkpoint, path)
            
            # Keep only last 3 checkpoints
            self._cleanup_old_checkpoints()
    
    def load_latest(self):
        checkpoints = sorted(self.dir.glob('checkpoint_*.pt'))
        if checkpoints:
            return torch.load(checkpoints[-1])
        return None
```

---

## 6. Performance Optimization

### 6.1 Profiling

```python
import time
from dataclasses import dataclass

@dataclass
class RoundProfile:
    round_num: int
    client_selection_time: float
    broadcast_time: float
    local_training_time: float
    aggregation_time: float
    total_time: float

class FLProfiler:
    def __init__(self):
        self.profiles = []
    
    def profile_round(self, round_fn, round_num):
        t0 = time.time()
        
        # Time each phase
        t_select = time.time()
        clients = select_clients()
        t_select = time.time() - t_select
        
        t_broadcast = time.time()
        broadcast_model(clients)
        t_broadcast = time.time() - t_broadcast
        
        t_train = time.time()
        updates = train_clients(clients)
        t_train = time.time() - t_train
        
        t_agg = time.time()
        aggregate(updates)
        t_agg = time.time() - t_agg
        
        profile = RoundProfile(
            round_num=round_num,
            client_selection_time=t_select,
            broadcast_time=t_broadcast,
            local_training_time=t_train,
            aggregation_time=t_agg,
            total_time=time.time() - t0
        )
        self.profiles.append(profile)
        return profile
```

### 6.2 GPU Utilization

Monitor and optimize GPU usage:

```python
import subprocess

def get_gpu_utilization():
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total',
         '--format=csv,noheader,nounits'],
        capture_output=True, text=True
    )
    lines = result.stdout.strip().split('\n')
    utils = []
    for line in lines:
        util, mem_used, mem_total = map(float, line.split(','))
        utils.append({
            'gpu_util': util,
            'mem_util': mem_used / mem_total * 100
        })
    return utils
```

### 6.3 Memory Optimization

For large-scale experiments:

```python
# Gradient accumulation for memory-constrained clients
def train_with_accumulation(model, dataloader, accumulation_steps=4):
    optimizer.zero_grad()
    
    for i, (x, y) in enumerate(dataloader):
        output = model(x)
        loss = criterion(output, y) / accumulation_steps
        loss.backward()
        
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
```

---

## 7. Exascale Considerations (Frontier-class)

### 7.1 Scale Targets

| Metric | Typical HPC | Frontier-class |
|--------|-------------|----------------|
| Nodes | 100s | 9,000+ |
| GPUs | 400 | 37,000+ |
| Interconnect | 100 Gbps | 50 TB/s aggregate |
| Memory | 10 TB | 9+ PB |

### 7.2 Challenges at Exascale

1. **Fault tolerance**: Node failures during long jobs
2. **Load balancing**: Heterogeneous node performance
3. **Communication bottlenecks**: Even high-bandwidth can saturate
4. **Debugging**: Distributed debugging at scale

### 7.3 Strategies

```python
# Fault-tolerant FL with checkpointing
class FaultTolerantFL:
    def __init__(self, checkpoint_freq=10):
        self.checkpoint_freq = checkpoint_freq
        
    def run(self, num_rounds):
        start_round = self.load_checkpoint() or 0
        
        for round_num in range(start_round, num_rounds):
            try:
                self.run_round(round_num)
                
                if round_num % self.checkpoint_freq == 0:
                    self.save_checkpoint(round_num)
                    
            except NodeFailureException as e:
                # Recover and retry without failed node
                self.exclude_node(e.node_id)
                self.run_round(round_num)
```

---

## 8. Recommendations

### 8.1 For Different Scales

| Scale | Recommendation |
|-------|----------------|
| **Development** (1-10 clients) | Single node, Flower simulation |
| **Small** (10-100 clients) | Multi-node, SLURM array jobs |
| **Medium** (100-1000 clients) | Dedicated cluster allocation |
| **Large** (1000+ clients) | Hierarchical, async aggregation |

### 8.2 Checklist for HPC FL

- [ ] Profile communication vs compute time
- [ ] Implement checkpointing for fault tolerance
- [ ] Use compression for bandwidth-limited scenarios
- [ ] Consider partial participation to reduce waiting
- [ ] Monitor GPU utilization and optimize batching
- [ ] Test at small scale before full allocation

---

## References

1. Bonawitz, K., et al. (2019). *Towards Federated Learning at Scale: A System Design*. MLSys.
2. Lai, F., et al. (2022). *FedScale: Benchmarking Model and System Heterogeneity in Federated Learning*. ICML.
3. ORNL. (2022). *Frontier User Guide*. https://docs.olcf.ornl.gov/systems/frontier_user_guide.html
