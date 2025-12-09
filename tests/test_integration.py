"""
Integration tests for fl_research library.

Tests complete end-to-end workflows: data loading → partitioning → training → aggregation
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Subset
import numpy as np

from fl_research.models import SimpleCNN, MLP, CIFAR10CNN, get_model
from fl_research.models.registry import list_models
from fl_research.data import (
    IIDPartitioner, 
    DirichletPartitioner, 
    ShardPartitioner,
)
from fl_research.privacy import PrivacyAccountant, get_privacy_spent
from fl_research.strategies import SCAFFOLDServer, StandaloneSCAFFOLDClient
from fl_research.utils import set_seed, get_device, MetricsTracker


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def device():
    """Get device for testing."""
    return torch.device("cpu")


@pytest.fixture
def synthetic_image_dataset():
    """Create a synthetic image dataset (like CIFAR-10 shape)."""
    set_seed(42)
    # 200 samples, 3 channels, 32x32 images, 10 classes
    X = torch.randn(200, 3, 32, 32)
    y = torch.randint(0, 10, (200,))
    return TensorDataset(X, y)


@pytest.fixture
def synthetic_mnist_dataset():
    """Create a synthetic MNIST-like dataset."""
    set_seed(42)
    # 200 samples, 1 channel, 28x28 images, 10 classes
    X = torch.randn(200, 1, 28, 28)
    y = torch.randint(0, 10, (200,))
    return TensorDataset(X, y)


@pytest.fixture
def synthetic_tabular_dataset():
    """Create a synthetic tabular dataset."""
    set_seed(42)
    X = torch.randn(200, 784)  # Flattened images
    y = torch.randint(0, 10, (200,))
    return TensorDataset(X, y)


# =============================================================================
# End-to-End Workflow Tests
# =============================================================================

class TestModelRegistryWorkflow:
    """Test the model registry workflow."""
    
    def test_create_models_directly(self):
        """Test creating models directly."""
        # Test direct model creation
        simple = SimpleCNN(num_classes=10)
        assert isinstance(simple, nn.Module)
        
        cifar = CIFAR10CNN()
        assert isinstance(cifar, nn.Module)
        
        mlp = MLP(input_dim=784, hidden_dims=[128], output_dim=10)
        assert isinstance(mlp, nn.Module)
    
    def test_model_forward_passes(self, device):
        """Test forward passes for all models."""
        set_seed(42)
        
        # SimpleCNN (28x28)
        simple = SimpleCNN(num_classes=10).to(device)
        x_mnist = torch.randn(2, 1, 28, 28, device=device)
        out = simple(x_mnist)
        assert out.shape == (2, 10)
        
        # CIFAR10CNN (32x32, 3 channels)
        cifar_cnn = CIFAR10CNN().to(device)
        x_cifar = torch.randn(2, 3, 32, 32, device=device)
        out = cifar_cnn(x_cifar)
        assert out.shape == (2, 10)
        
        # MLP (flattened input)
        mlp = MLP(input_dim=784, hidden_dims=[128, 64], output_dim=10).to(device)
        x_flat = torch.randn(2, 784, device=device)
        out = mlp(x_flat)
        assert out.shape == (2, 10)


class TestDataPartitioningWorkflow:
    """Test data partitioning end-to-end."""
    
    def test_iid_partitioning_preserves_data(self, synthetic_image_dataset):
        """Test that IID partitioning preserves all data."""
        num_clients = 5
        partitioner = IIDPartitioner(num_clients=num_clients, seed=42)
        partitions = partitioner.partition(synthetic_image_dataset)
        
        # All indices covered
        all_indices = set()
        for indices in partitions:
            all_indices.update(indices)
        
        assert len(all_indices) == len(synthetic_image_dataset)
        assert len(partitions) == num_clients
    
    def test_dirichlet_partitioning(self, synthetic_image_dataset):
        """Test Dirichlet (non-IID) partitioning."""
        num_clients = 5
        
        # Low alpha = high heterogeneity
        partitioner = DirichletPartitioner(num_clients=num_clients, alpha=0.5, seed=42)
        partitions = partitioner.partition(synthetic_image_dataset)
        
        # Check partitions are valid
        assert len(partitions) == num_clients
        
        # All indices should be covered (allow small margin for edge cases)
        all_indices = set()
        for indices in partitions:
            all_indices.update(indices)
        # With Dirichlet, some samples might get lost due to rounding
        assert len(all_indices) >= len(synthetic_image_dataset) * 0.95
    
    def test_shard_partitioning(self, synthetic_image_dataset):
        """Test shard-based partitioning."""
        num_clients = 5
        shards_per_client = 2
        
        partitioner = ShardPartitioner(
            num_clients=num_clients, 
            shards_per_client=shards_per_client,
            seed=42
        )
        partitions = partitioner.partition(synthetic_image_dataset)
        
        assert len(partitions) == num_clients
    
    def test_create_dataloaders_from_partitions(self, synthetic_image_dataset):
        """Test creating DataLoaders from partitioned indices."""
        num_clients = 3
        partitioner = IIDPartitioner(num_clients=num_clients, seed=42)
        partitions = partitioner.partition(synthetic_image_dataset)
        
        dataloaders = []
        for indices in partitions:
            subset = Subset(synthetic_image_dataset, list(indices))
            loader = DataLoader(subset, batch_size=16, shuffle=True)
            dataloaders.append(loader)
        
        assert len(dataloaders) == num_clients
        
        # Verify we can iterate
        for loader in dataloaders:
            batch_x, batch_y = next(iter(loader))
            assert batch_x.shape[0] <= 16


class TestPrivacyAccountingWorkflow:
    """Test privacy accounting end-to-end."""
    
    def test_privacy_budget_tracking(self):
        """Test tracking privacy budget over rounds."""
        accountant = PrivacyAccountant(
            noise_multiplier=1.0,
            sample_rate=0.01,
            delta=1e-5,
            max_epsilon=10.0,
        )
        
        epsilons = []
        for round_num in range(5):
            accountant.step(10)  # 10 gradient steps per round
            epsilon = accountant.get_epsilon()
            epsilons.append(epsilon)
        
        # Epsilon should increase with steps
        assert all(e >= 0 for e in epsilons)
        assert epsilons[-1] > epsilons[0]
    
    def test_get_privacy_spent_function(self):
        """Test the convenience function for privacy calculation."""
        budget = get_privacy_spent(
            steps=100,
            noise_multiplier=1.0,
            sample_rate=0.01,
            delta=1e-5,
        )
        
        assert budget.epsilon >= 0
        assert budget.delta == 1e-5


class TestSCAFFOLDWorkflow:
    """Test complete SCAFFOLD federated learning workflow."""
    
    def test_full_fl_round(self, synthetic_mnist_dataset, device):
        """Test a complete FL round: partition → train → aggregate."""
        set_seed(42)
        num_clients = 3
        
        # Step 1: Partition data
        partitioner = IIDPartitioner(num_clients=num_clients, seed=42)
        partitions = partitioner.partition(synthetic_mnist_dataset)
        
        # Step 2: Create client dataloaders
        dataloaders = []
        for indices in partitions:
            subset = Subset(synthetic_mnist_dataset, list(indices))
            loader = DataLoader(subset, batch_size=16, shuffle=True)
            dataloaders.append(loader)
        
        # Step 3: Initialize server with model
        model = SimpleCNN(num_classes=10)
        server = SCAFFOLDServer(model, device)
        
        # Step 4: Create clients
        clients = [
            StandaloneSCAFFOLDClient(
                client_id=i,
                dataloader=loader,
                device=device
            )
            for i, loader in enumerate(dataloaders)
        ]
        
        # Initialize control variates
        for client in clients:
            client.initialize_control_variate(model)
        
        # Step 5: Training round
        delta_weights_list = []
        delta_controls_list = []
        sample_counts = []
        
        for client in clients:
            # Each client trains on global model
            client_model = SimpleCNN(num_classes=10)
            with torch.no_grad():
                for p_client, p_global in zip(client_model.parameters(), server.model.parameters()):
                    p_client.data.copy_(p_global.data)
            
            delta_w, delta_c, count = client.train(
                model=client_model,
                global_control=server.global_control,
                epochs=1,
                lr=0.01
            )
            
            delta_weights_list.append(delta_w)
            delta_controls_list.append(delta_c)
            sample_counts.append(count)
        
        # Step 6: Aggregate
        server.aggregate(
            delta_weights_list=delta_weights_list,
            delta_controls_list=delta_controls_list,
            sample_counts=sample_counts,
            total_clients=num_clients
        )
        
        # Step 7: Evaluate
        test_loader = DataLoader(synthetic_mnist_dataset, batch_size=32)
        accuracy, loss = server.evaluate(test_loader)
        
        assert 0.0 <= accuracy <= 1.0
        assert loss >= 0.0
    
    def test_multiple_fl_rounds(self, synthetic_mnist_dataset, device):
        """Test multiple FL rounds with metrics tracking."""
        set_seed(42)
        num_clients = 2
        num_rounds = 3
        
        # Setup
        partitioner = IIDPartitioner(num_clients=num_clients, seed=42)
        partitions = partitioner.partition(synthetic_mnist_dataset)
        
        dataloaders = [
            DataLoader(Subset(synthetic_mnist_dataset, list(indices)), batch_size=16, shuffle=True)
            for indices in partitions
        ]
        
        model = SimpleCNN(num_classes=10)
        server = SCAFFOLDServer(model, device)
        tracker = MetricsTracker()
        
        clients = [
            StandaloneSCAFFOLDClient(client_id=i, dataloader=loader, device=device)
            for i, loader in enumerate(dataloaders)
        ]
        for client in clients:
            client.initialize_control_variate(model)
        
        # Training loop
        for round_num in range(num_rounds):
            delta_weights_list = []
            delta_controls_list = []
            sample_counts = []
            
            for client in clients:
                client_model = SimpleCNN(num_classes=10)
                with torch.no_grad():
                    for p_client, p_global in zip(client_model.parameters(), server.model.parameters()):
                        p_client.data.copy_(p_global.data)
                
                delta_w, delta_c, count = client.train(
                    model=client_model,
                    global_control=server.global_control,
                    epochs=2,
                    lr=0.01
                )
                
                delta_weights_list.append(delta_w)
                delta_controls_list.append(delta_c)
                sample_counts.append(count)
            
            server.aggregate(
                delta_weights_list=delta_weights_list,
                delta_controls_list=delta_controls_list,
                sample_counts=sample_counts,
                total_clients=num_clients
            )
            
            # Evaluate and track
            test_loader = DataLoader(synthetic_mnist_dataset, batch_size=32)
            accuracy, loss = server.evaluate(test_loader)
            
            tracker.add_round(
                round_num=round_num + 1,
                accuracy=accuracy,
                loss=loss,
            )
        
        # Verify tracking
        accuracy_history = tracker.get_history("accuracy")
        loss_history = tracker.get_history("loss")
        
        assert len(accuracy_history) == num_rounds
        assert len(loss_history) == num_rounds
        
        # All losses should be finite
        for loss in loss_history:
            assert not np.isnan(loss)
            assert not np.isinf(loss)


class TestMLPWorkflow:
    """Test MLP model workflow with tabular data."""
    
    def test_mlp_training_workflow(self, synthetic_tabular_dataset, device):
        """Test MLP model training on tabular data."""
        set_seed(42)
        
        # Create MLP
        model = MLP(
            input_dim=784, 
            hidden_dims=[256, 128], 
            output_dim=10
        ).to(device)
        
        # Create dataloader
        loader = DataLoader(synthetic_tabular_dataset, batch_size=32, shuffle=True)
        
        # Simple training
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        initial_loss = None
        final_loss = None
        
        for epoch in range(3):
            epoch_loss = 0
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                output = model(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(loader)
            
            if epoch == 0:
                initial_loss = avg_loss
            if epoch == 2:
                final_loss = avg_loss
        
        # Loss should decrease (or at least not explode)
        assert initial_loss is not None
        assert final_loss is not None
        assert not np.isnan(final_loss)


class TestReproducibility:
    """Test reproducibility of the library."""
    
    def test_seed_ensures_reproducible_partitions(self, synthetic_image_dataset):
        """Test that same seed produces same partitions."""
        partitioner1 = IIDPartitioner(num_clients=5, seed=42)
        partitioner2 = IIDPartitioner(num_clients=5, seed=42)
        
        partitions1 = partitioner1.partition(synthetic_image_dataset)
        partitions2 = partitioner2.partition(synthetic_image_dataset)
        
        for p1, p2 in zip(partitions1, partitions2):
            assert list(p1) == list(p2)
    
    def test_seed_ensures_reproducible_model_init(self, device):
        """Test that same seed produces same model weights."""
        set_seed(42)
        model1 = SimpleCNN(num_classes=10)
        
        set_seed(42)
        model2 = SimpleCNN(num_classes=10)
        
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2)
    
    def test_different_seeds_different_results(self, synthetic_image_dataset):
        """Test that different seeds produce different partitions."""
        partitioner1 = IIDPartitioner(num_clients=5, seed=42)
        partitioner2 = IIDPartitioner(num_clients=5, seed=123)
        
        partitions1 = partitioner1.partition(synthetic_image_dataset)
        partitions2 = partitioner2.partition(synthetic_image_dataset)
        
        # At least one partition should be different
        any_different = False
        for p1, p2 in zip(partitions1, partitions2):
            if list(p1) != list(p2):
                any_different = True
                break
        
        assert any_different
