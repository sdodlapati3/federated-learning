#!/usr/bin/env python3
"""
Minimal Example: Federated Learning with SCAFFOLD
==================================================

This example demonstrates the core FL workflow using the fl_research library:
1. Load and partition data across clients
2. Initialize server and clients with SCAFFOLD algorithm
3. Run federated training rounds
4. Track and report results

Run with: python examples/minimal_example.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset

from fl_research.models import SimpleCNN
from fl_research.data import DirichletPartitioner
from fl_research.strategies import SCAFFOLDServer, StandaloneSCAFFOLDClient
from fl_research.utils import set_seed, get_device, MetricsTracker


def create_synthetic_data(num_samples: int = 500, num_classes: int = 10):
    """Create synthetic MNIST-like data for demonstration."""
    X = torch.randn(num_samples, 1, 28, 28)
    y = torch.randint(0, num_classes, (num_samples,))
    return TensorDataset(X, y)


def main():
    # Configuration
    NUM_CLIENTS = 5
    NUM_ROUNDS = 5
    LOCAL_EPOCHS = 2
    LEARNING_RATE = 0.01
    ALPHA = 0.5  # Dirichlet concentration (lower = more heterogeneous)
    
    print("=" * 60)
    print("FL Research Library - Minimal SCAFFOLD Example")
    print("=" * 60)
    
    # Set seed for reproducibility
    set_seed(42)
    device = get_device()
    print(f"\nDevice: {device}")
    
    # Step 1: Create and partition data
    print(f"\n[1/4] Creating synthetic dataset and partitioning...")
    dataset = create_synthetic_data(num_samples=500)
    
    partitioner = DirichletPartitioner(
        num_clients=NUM_CLIENTS, 
        alpha=ALPHA, 
        seed=42
    )
    partitions = partitioner.partition(dataset)
    
    print(f"  - {len(dataset)} total samples")
    print(f"  - {NUM_CLIENTS} clients")
    print(f"  - Partition sizes: {[len(p) for p in partitions]}")
    
    # Create dataloaders for each client
    dataloaders = []
    for indices in partitions:
        subset = Subset(dataset, list(indices))
        loader = DataLoader(subset, batch_size=32, shuffle=True)
        dataloaders.append(loader)
    
    # Step 2: Initialize model, server, and clients
    print(f"\n[2/4] Initializing SCAFFOLD server and {NUM_CLIENTS} clients...")
    model = SimpleCNN(num_classes=10)
    server = SCAFFOLDServer(model, device)
    
    clients = []
    for i, loader in enumerate(dataloaders):
        client = StandaloneSCAFFOLDClient(
            client_id=i,
            dataloader=loader,
            device=device
        )
        client.initialize_control_variate(model)
        clients.append(client)
    
    tracker = MetricsTracker(experiment_name="scaffold_demo")
    
    # Create test loader (use full dataset for simplicity)
    test_loader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    # Step 3: Federated training loop
    print(f"\n[3/4] Starting federated training ({NUM_ROUNDS} rounds)...")
    print("-" * 60)
    
    for round_num in range(1, NUM_ROUNDS + 1):
        # Collect updates from all clients
        delta_weights_list = []
        delta_controls_list = []
        sample_counts = []
        
        global_weights = server.get_weights()
        
        for client in clients:
            # Create fresh model with global weights
            client_model = SimpleCNN(num_classes=10).to(device)
            with torch.no_grad():
                for p_client, w in zip(client_model.parameters(), global_weights):
                    p_client.data.copy_(w.to(device))
            
            # Local training with SCAFFOLD correction
            delta_w, delta_c, count = client.train(
                model=client_model,
                global_control=server.global_control,
                epochs=LOCAL_EPOCHS,
                lr=LEARNING_RATE
            )
            
            delta_weights_list.append(delta_w)
            delta_controls_list.append(delta_c)
            sample_counts.append(count)
        
        # Aggregate updates on server
        server.aggregate(
            delta_weights_list=delta_weights_list,
            delta_controls_list=delta_controls_list,
            sample_counts=sample_counts,
            total_clients=NUM_CLIENTS
        )
        
        # Evaluate global model
        accuracy, loss = server.evaluate(test_loader)
        
        # Track metrics
        tracker.add_round(
            round_num=round_num,
            accuracy=accuracy,
            loss=loss,
            num_clients=NUM_CLIENTS,
            num_samples=sum(sample_counts)
        )
        
        print(f"Round {round_num:2d}: Accuracy = {accuracy:.4f}, Loss = {loss:.4f}")
    
    # Step 4: Report results
    print("-" * 60)
    print(f"\n[4/4] Training complete!")
    
    best_round, best_acc = tracker.get_best("accuracy")
    final = tracker.get_final()
    
    print(f"\nResults:")
    print(f"  - Best accuracy: {best_acc:.4f} (round {best_round})")
    print(f"  - Final accuracy: {final.accuracy:.4f}")
    print(f"  - Final loss: {final.loss:.4f}")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
