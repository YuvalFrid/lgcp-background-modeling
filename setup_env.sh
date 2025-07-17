#!/bin/bash

# Create virtual environment
python3 -m venv .venv

# Activate it (manual step if running from terminal)
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install numpy matplotlib torch tqdm

# Save exact versions
pip freeze > requirements.txt

echo "✅ Environment setup complete. Activate it with:"
echo "source .venv/bin/activate"

