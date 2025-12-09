#!/bin/bash
# Flower Federated Learning Environment Setup Script

set -e

echo "🌸 Setting up Flower Federated Learning environment..."

# Create virtual environment
if [ ! -d "flower-env" ]; then
    echo "📦 Creating Python virtual environment..."
    python -m venv flower-env
else
    echo "✓ Virtual environment already exists"
fi

# Activate environment
echo "🔧 Activating environment..."
source flower-env/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install Flower with simulation support
echo "🌼 Installing Flower framework..."
pip install -U "flwr[simulation]"

# Install PyTorch (CPU version for tutorial)
echo "🔥 Installing PyTorch..."
pip install torch torchvision

# Verify installation
echo ""
echo "✅ Installation complete!"
echo ""
echo "Flower version:"
pip show flwr | grep Version
echo ""
echo "PyTorch version:"
python -c "import torch; print(torch.__version__)"
echo ""
echo "🎯 Next steps:"
echo "   1. Activate environment: source flower-env/bin/activate"
echo "   2. Create Flower project: flwr new flower-tutorial --framework pytorch --username flwrlabs"
echo "   3. Install project: cd flower-tutorial && pip install -e ."
echo "   4. Run simulation: flwr run ."
