#!/bin/bash
"""
Minimal Installation Verification Example
 
This script provides a simple working example to verify that all dependencies 
are correctly installed and that the Semi-Supervised Rotation Forest library 
is functioning properly.
 
IMPORTANT: This is NOT intended as a performance benchmark or demonstration of 
the method's effectiveness. The dataset used here is deliberately minimal and 
may not represent scenarios where Semi-Supervised Rotation Forest excels.
 
For comprehensive performance comparisons and proper experimental setup, 
please refer to the experiments described in the paper and the full 
experimental scripts in experiments.py.
 
Purpose:
- Verify successful installation of all required packages
- Confirm basic functionality of the library
- Provide a quick sanity check that the code runs without errors
"""

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
