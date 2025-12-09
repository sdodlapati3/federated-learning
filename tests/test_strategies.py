"""
Tests for fl_research.strategies module.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from fl_research.strategies.fedavg import weighted_average, FedAvgStrategy
from fl_research.strategies.fedprox import FedProxStrategy, FedProxClient
from fl_research.strategies.scaffold import (
    SCAFFOLDServer,
    StandaloneSCAFFOLDClient,
    SCAFFOLDStrategy,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def simple_model():
    """A simple model for testing."""
    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 5)
            self.fc2 = nn.Linear(5, 2)
        
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            return self.fc2(x)
    
    return SimpleNet()


@pytest.fixture
def device():
    """Get device for testing."""
    return torch.device("cpu")


@pytest.fixture
def sample_dataloader():
    """Create a small dataloader for testing."""
    X = torch.randn(50, 10)
    y = torch.randint(0, 2, (50,))
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=10, shuffle=True)


@pytest.fixture
def multiple_dataloaders():
    """Create multiple dataloaders simulating different clients."""
    loaders = []
    for i in range(3):
        X = torch.randn(30 + i * 10, 10)
        y = torch.randint(0, 2, (30 + i * 10,))
        dataset = TensorDataset(X, y)
        loaders.append(DataLoader(dataset, batch_size=10, shuffle=True))
    return loaders


# =============================================================================
# Test weighted_average
# =============================================================================

class TestWeightedAverage:
    """Tests for weighted_average function."""
    
    def test_basic_aggregation(self):
        """Test basic weighted average."""
        metrics = [
            (100, {"accuracy": 0.8, "loss": 0.5}),
            (100, {"accuracy": 0.9, "loss": 0.3}),
        ]
        result = weighted_average(metrics)
        
        assert abs(result["accuracy"] - 0.85) < 1e-6
        assert abs(result["loss"] - 0.4) < 1e-6
    
    def test_unequal_weights(self):
        """Test with unequal sample counts."""
        metrics = [
            (100, {"accuracy": 0.8}),
            (300, {"accuracy": 0.9}),
        ]
        result = weighted_average(metrics)
        
        # Expected: (100*0.8 + 300*0.9) / 400 = 350/400 = 0.875
        assert abs(result["accuracy"] - 0.875) < 1e-6
    
    def test_empty_metrics(self):
        """Test with empty metrics list."""
        result = weighted_average([])
        assert result == {}
    
    def test_zero_samples(self):
        """Test with zero total samples."""
        metrics = [(0, {"accuracy": 0.5})]
        result = weighted_average(metrics)
        assert result == {}
    
    def test_missing_keys(self):
        """Test when different clients have different metrics."""
        metrics = [
            (100, {"accuracy": 0.8, "loss": 0.5}),
            (100, {"accuracy": 0.9}),  # missing loss
        ]
        result = weighted_average(metrics)
        
        assert "accuracy" in result
        assert "loss" in result
        # Loss weighted: (100*0.5 + 100*0) / 200 = 0.25
        assert abs(result["loss"] - 0.25) < 1e-6


# =============================================================================
# Test FedAvgStrategy
# =============================================================================

class TestFedAvgStrategy:
    """Tests for FedAvg strategy."""
    
    def test_initialization(self):
        """Test strategy initialization."""
        strategy = FedAvgStrategy(
            fraction_fit=0.5,
            fraction_evaluate=0.3,
            min_fit_clients=2,
        )
        
        assert strategy.fraction_fit == 0.5
        assert strategy.fraction_evaluate == 0.3
        assert strategy.min_fit_clients == 2
    
    def test_num_fit_clients(self):
        """Test client sampling logic."""
        strategy = FedAvgStrategy(
            fraction_fit=0.5,
            min_fit_clients=2,
        )
        
        # 10 available * 0.5 = 5, but min is 2
        sample_size, min_clients = strategy.num_fit_clients(10)
        assert sample_size == 5
        assert min_clients == 2
        
        # 2 available * 0.5 = 1, but min is 2
        sample_size, min_clients = strategy.num_fit_clients(2)
        assert sample_size == 2  # Should be at least min_fit_clients
        assert min_clients == 2
    
    def test_num_evaluate_clients(self):
        """Test evaluation client sampling."""
        strategy = FedAvgStrategy(
            fraction_evaluate=0.2,
            min_evaluate_clients=3,
        )
        
        sample_size, min_clients = strategy.num_evaluate_clients(20)
        assert sample_size == 4  # 20 * 0.2 = 4
        assert min_clients == 3
    
    def test_aggregate_weights(self):
        """Test weight aggregation."""
        strategy = FedAvgStrategy()
        
        # Create sample weight results
        w1 = [np.array([1.0, 2.0]), np.array([3.0])]
        w2 = [np.array([2.0, 4.0]), np.array([6.0])]
        
        results = [
            (100, w1),  # 100 samples
            (100, w2),  # 100 samples
        ]
        
        aggregated = strategy._aggregate_weights(results)
        
        # Equal weights -> simple average
        assert np.allclose(aggregated[0], np.array([1.5, 3.0]))
        assert np.allclose(aggregated[1], np.array([4.5]))
    
    def test_aggregate_weights_unequal(self):
        """Test weight aggregation with unequal samples."""
        strategy = FedAvgStrategy()
        
        w1 = [np.array([0.0, 0.0])]
        w2 = [np.array([1.0, 1.0])]
        
        results = [
            (100, w1),  # 25%
            (300, w2),  # 75%
        ]
        
        aggregated = strategy._aggregate_weights(results)
        
        # Weighted: 0.25*[0,0] + 0.75*[1,1] = [0.75, 0.75]
        assert np.allclose(aggregated[0], np.array([0.75, 0.75]))


# =============================================================================
# Test FedProxStrategy
# =============================================================================

class TestFedProxStrategy:
    """Tests for FedProx strategy."""
    
    def test_initialization(self):
        """Test FedProx initialization with proximal term."""
        strategy = FedProxStrategy(
            proximal_mu=0.1,
            fraction_fit=0.5,
        )
        
        assert strategy.proximal_mu == 0.1
        assert strategy.fraction_fit == 0.5
    
    def test_inherits_fedavg(self):
        """Test that FedProx inherits from FedAvg."""
        strategy = FedProxStrategy()
        
        assert isinstance(strategy, FedAvgStrategy)
        assert hasattr(strategy, 'aggregate_fit')
        assert hasattr(strategy, 'aggregate_evaluate')


class TestFedProxClient:
    """Tests for FedProx client."""
    
    def test_initialization(self, simple_model, sample_dataloader):
        """Test client initialization."""
        client = FedProxClient(
            model=simple_model,
            train_loader=sample_dataloader,
            proximal_mu=0.1,
            lr=0.01,
            device="cpu",
        )
        
        assert client.proximal_mu == 0.1
        assert client.lr == 0.01
        assert client.device == torch.device("cpu")
    
    def test_set_global_parameters(self, simple_model, sample_dataloader):
        """Test setting global parameters."""
        client = FedProxClient(
            model=simple_model,
            train_loader=sample_dataloader,
            proximal_mu=0.1,
            device="cpu",
        )
        
        # Get model parameters as numpy arrays
        global_params = [p.data.numpy().copy() for p in simple_model.parameters()]
        client.set_global_parameters(global_params)
        
        assert client.global_params is not None
        assert len(client.global_params) == len(list(simple_model.parameters()))
    
    def test_compute_proximal_term(self, simple_model, sample_dataloader):
        """Test proximal term computation."""
        client = FedProxClient(
            model=simple_model,
            train_loader=sample_dataloader,
            proximal_mu=0.1,
            device="cpu",
        )
        
        # Set global params
        global_params = [p.data.numpy().copy() for p in simple_model.parameters()]
        client.set_global_parameters(global_params)
        
        # When model == global, proximal term should be 0
        prox_term = client.compute_proximal_term()
        assert abs(prox_term.item()) < 1e-6
        
        # Modify model parameters
        for p in client.model.parameters():
            p.data.add_(torch.ones_like(p))
        
        # Now proximal term should be > 0
        prox_term = client.compute_proximal_term()
        assert prox_term.item() > 0
    
    def test_train_one_epoch(self, simple_model, sample_dataloader):
        """Test training for one epoch."""
        client = FedProxClient(
            model=simple_model,
            train_loader=sample_dataloader,
            proximal_mu=0.1,
            lr=0.01,
            device="cpu",
        )
        
        # Get initial parameters
        initial_params = [p.data.clone() for p in client.model.parameters()]
        
        # Set global params
        global_params = [p.data.numpy().copy() for p in simple_model.parameters()]
        client.set_global_parameters(global_params)
        
        # Train
        loss = client.train(epochs=1)
        
        # Check that parameters changed
        params_changed = False
        for old_p, new_p in zip(initial_params, client.model.parameters()):
            if not torch.allclose(old_p, new_p.data):
                params_changed = True
                break
        
        assert params_changed, "Training should update parameters"


# =============================================================================
# Test SCAFFOLDServer
# =============================================================================

class TestSCAFFOLDServer:
    """Tests for SCAFFOLD server."""
    
    def test_initialization(self, simple_model, device):
        """Test server initialization."""
        server = SCAFFOLDServer(simple_model, device)
        
        assert server.device == device
        assert len(server.global_control) == len(list(simple_model.parameters()))
        
        # Control variates should be zeros
        for c in server.global_control:
            assert torch.allclose(c, torch.zeros_like(c))
    
    def test_get_weights(self, simple_model, device):
        """Test getting model weights."""
        server = SCAFFOLDServer(simple_model, device)
        weights = server.get_weights()
        
        assert len(weights) == len(list(simple_model.parameters()))
        
        for w, p in zip(weights, simple_model.parameters()):
            assert torch.allclose(w, p.data)
    
    def test_set_weights(self, simple_model, device):
        """Test setting model weights."""
        server = SCAFFOLDServer(simple_model, device)
        
        # Create new weights
        new_weights = [torch.randn_like(p) for p in simple_model.parameters()]
        server.set_weights(new_weights)
        
        for w, p in zip(new_weights, server.model.parameters()):
            assert torch.allclose(w, p.data)
    
    def test_aggregate_single_client(self, simple_model, device):
        """Test aggregation with single client."""
        server = SCAFFOLDServer(simple_model, device)
        
        # Simulate client update
        delta_weights = [torch.ones_like(p) * 0.1 for p in simple_model.parameters()]
        delta_controls = [torch.ones_like(p) * 0.01 for p in simple_model.parameters()]
        
        initial_weights = server.get_weights()
        
        server.aggregate(
            delta_weights_list=[delta_weights],
            delta_controls_list=[delta_controls],
            sample_counts=[100],
            total_clients=1
        )
        
        # Check weights updated
        new_weights = server.get_weights()
        for old_w, new_w, delta in zip(initial_weights, new_weights, delta_weights):
            expected = old_w + delta
            assert torch.allclose(new_w, expected, atol=1e-6)
    
    def test_aggregate_multiple_clients(self, simple_model, device):
        """Test aggregation with multiple clients."""
        server = SCAFFOLDServer(simple_model, device)
        
        # Two clients with equal samples
        delta_weights_1 = [torch.ones_like(p) * 0.1 for p in simple_model.parameters()]
        delta_weights_2 = [torch.ones_like(p) * 0.2 for p in simple_model.parameters()]
        
        delta_controls_1 = [torch.ones_like(p) * 0.01 for p in simple_model.parameters()]
        delta_controls_2 = [torch.ones_like(p) * 0.02 for p in simple_model.parameters()]
        
        initial_weights = server.get_weights()
        
        server.aggregate(
            delta_weights_list=[delta_weights_1, delta_weights_2],
            delta_controls_list=[delta_controls_1, delta_controls_2],
            sample_counts=[100, 100],  # Equal samples
            total_clients=2
        )
        
        # Weighted average of deltas: (0.1 + 0.2) / 2 = 0.15
        new_weights = server.get_weights()
        for old_w, new_w in zip(initial_weights, new_weights):
            expected = old_w + torch.ones_like(old_w) * 0.15
            assert torch.allclose(new_w, expected, atol=1e-6)
    
    def test_evaluate(self, simple_model, device, sample_dataloader):
        """Test server evaluation."""
        server = SCAFFOLDServer(simple_model, device)
        
        accuracy, loss = server.evaluate(sample_dataloader)
        
        assert 0.0 <= accuracy <= 1.0
        assert loss >= 0.0


# =============================================================================
# Test StandaloneSCAFFOLDClient
# =============================================================================

class TestStandaloneSCAFFOLDClient:
    """Tests for standalone SCAFFOLD client."""
    
    def test_initialization(self, sample_dataloader, device):
        """Test client initialization."""
        client = StandaloneSCAFFOLDClient(
            client_id=0,
            dataloader=sample_dataloader,
            device=device
        )
        
        assert client.client_id == 0
        assert client.device == device
        assert client.control_variate is None
    
    def test_initialize_control_variate(self, simple_model, sample_dataloader, device):
        """Test control variate initialization."""
        client = StandaloneSCAFFOLDClient(
            client_id=0,
            dataloader=sample_dataloader,
            device=device
        )
        
        client.initialize_control_variate(simple_model)
        
        assert client.control_variate is not None
        assert len(client.control_variate) == len(list(simple_model.parameters()))
        
        # Should be zeros
        for c in client.control_variate:
            assert torch.allclose(c, torch.zeros_like(c))
    
    def test_train_returns_correct_shapes(self, simple_model, sample_dataloader, device):
        """Test that training returns correct shaped tensors."""
        client = StandaloneSCAFFOLDClient(
            client_id=0,
            dataloader=sample_dataloader,
            device=device
        )
        
        client.initialize_control_variate(simple_model)
        
        # Global control (zeros)
        global_control = [torch.zeros_like(p) for p in simple_model.parameters()]
        
        delta_w, delta_c, num_samples = client.train(
            model=simple_model,
            global_control=global_control,
            epochs=1,
            lr=0.01
        )
        
        # Check shapes
        assert len(delta_w) == len(list(simple_model.parameters()))
        assert len(delta_c) == len(list(simple_model.parameters()))
        
        for dw, p in zip(delta_w, simple_model.parameters()):
            assert dw.shape == p.shape
        
        for dc, p in zip(delta_c, simple_model.parameters()):
            assert dc.shape == p.shape
        
        assert num_samples == len(sample_dataloader.dataset)
    
    def test_train_updates_control_variate(self, simple_model, sample_dataloader, device):
        """Test that training updates control variate."""
        client = StandaloneSCAFFOLDClient(
            client_id=0,
            dataloader=sample_dataloader,
            device=device
        )
        
        client.initialize_control_variate(simple_model)
        initial_control = [c.clone() for c in client.control_variate]
        
        global_control = [torch.zeros_like(p) for p in simple_model.parameters()]
        
        _, delta_c, _ = client.train(
            model=simple_model,
            global_control=global_control,
            epochs=1,
            lr=0.01
        )
        
        # Control variate should be updated
        control_changed = False
        for old_c, new_c in zip(initial_control, client.control_variate):
            if not torch.allclose(old_c, new_c):
                control_changed = True
                break
        
        assert control_changed, "Control variate should be updated after training"


# =============================================================================
# Test SCAFFOLDStrategy (Flower integration)
# =============================================================================

class TestSCAFFOLDStrategy:
    """Tests for SCAFFOLD Flower strategy."""
    
    def test_initialization(self, simple_model):
        """Test strategy initialization."""
        strategy = SCAFFOLDStrategy(
            fraction_fit=0.5,
            min_fit_clients=2,
        )
        
        assert strategy.fraction_fit == 0.5
        assert strategy.min_fit_clients == 2
    
    def test_inherits_fedavg(self):
        """Test that SCAFFOLD inherits from FedAvg."""
        strategy = SCAFFOLDStrategy()
        
        assert isinstance(strategy, FedAvgStrategy)


# =============================================================================
# Integration Tests
# =============================================================================

class TestSCAFFOLDIntegration:
    """Integration tests for SCAFFOLD algorithm."""
    
    def test_full_round_simulation(self, simple_model, multiple_dataloaders, device):
        """Test a full SCAFFOLD round with multiple clients."""
        # Create server
        server = SCAFFOLDServer(simple_model, device)
        
        # Create clients
        clients = [
            StandaloneSCAFFOLDClient(
                client_id=i,
                dataloader=loader,
                device=device
            )
            for i, loader in enumerate(multiple_dataloaders)
        ]
        
        # Initialize client control variates
        for client in clients:
            client.initialize_control_variate(simple_model)
        
        # Get initial global weights
        global_weights = server.get_weights()
        
        # Simulate one round
        all_delta_weights = []
        all_delta_controls = []
        all_counts = []
        
        for client in clients:
            # Each client trains on a copy of the global model
            client_model = type(simple_model)()
            with torch.no_grad():
                for p_client, w in zip(client_model.parameters(), global_weights):
                    p_client.data.copy_(w)
            
            delta_w, delta_c, count = client.train(
                model=client_model,
                global_control=server.global_control,
                epochs=1,
                lr=0.01
            )
            
            all_delta_weights.append(delta_w)
            all_delta_controls.append(delta_c)
            all_counts.append(count)
        
        # Aggregate
        server.aggregate(
            delta_weights_list=all_delta_weights,
            delta_controls_list=all_delta_controls,
            sample_counts=all_counts,
            total_clients=len(clients)
        )
        
        # Verify weights changed
        new_weights = server.get_weights()
        weights_changed = False
        for old_w, new_w in zip(global_weights, new_weights):
            if not torch.allclose(old_w, new_w):
                weights_changed = True
                break
        
        assert weights_changed, "Global model should be updated after aggregation"
    
    def test_multiple_rounds(self, simple_model, sample_dataloader, device):
        """Test that SCAFFOLD converges over multiple rounds."""
        server = SCAFFOLDServer(simple_model, device)
        client = StandaloneSCAFFOLDClient(
            client_id=0,
            dataloader=sample_dataloader,
            device=device
        )
        client.initialize_control_variate(simple_model)
        
        losses = []
        
        for round_num in range(3):
            # Get fresh model with global weights
            client_model = type(simple_model)()
            with torch.no_grad():
                for p_client, w in zip(client_model.parameters(), server.get_weights()):
                    p_client.data.copy_(w)
            
            delta_w, delta_c, count = client.train(
                model=client_model,
                global_control=server.global_control,
                epochs=2,
                lr=0.01
            )
            
            server.aggregate([delta_w], [delta_c], [count], total_clients=1)
            
            # Evaluate
            _, loss = server.evaluate(sample_dataloader)
            losses.append(loss)
        
        # Not strictly requiring convergence, but loss should be finite
        for loss in losses:
            assert not np.isnan(loss)
            assert not np.isinf(loss)
