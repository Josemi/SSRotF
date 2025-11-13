#!/bin/bash
git clone https://github.com/Josemi/SSRotF.git
conda create -n SSRotF python=3.10.13
conda activate SSRotF
cd SSRotF
pip install -r requirements.txt
python example.py
