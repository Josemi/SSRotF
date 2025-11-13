#!/bin/bash

# Display what will be executed
echo "=============================================="
echo "Semi-Supervised Rotation Forest - Setup Script"
echo "=============================================="
echo ""
echo "This script will:"
echo "  1. Clone the SSRotF repository from GitHub"
echo "  2. Create a conda environment (SSRotF, Python 3.10.13)"
echo "  3. Install required dependencies"
echo "  4. Run the installation verification example"
echo ""
echo "Note: The example script is only for verifying"
echo "      that dependencies are correctly installed."
echo ""
read -p "Do you want to proceed? (y/n): " -n 1 -r
echo ""
 
# Check user response
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Installation cancelled."
    exit 1
fi
 
echo ""
echo "Starting installation..."
echo ""
 
# Execute installation steps
git clone https://github.com/Josemi/SSRotF.git
 
conda create -n SSRotF python=3.10.13 -y
 
# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate SSRotF
 
cd SSRotF
pip install -r requirements.txt
 
echo ""
echo "=============================================="
echo "Installation complete!"
echo "=============================================="
echo ""
read -p "Press Enter to run the installation verification example..." 
echo ""
 
# Clear screen before running example
clear
 
echo "=============================================="
echo "Running installation verification example..."
echo "=============================================="
echo ""
python example.py
 
echo ""
echo "Setup complete!"
