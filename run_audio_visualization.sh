#!/bin/bash

# Create and activate virtual environment
python3 -m venv my_audio_venv
source my_audio_venv/bin/activate

# Install dependencies
pip install Cython numpy

# Debug information
which cython
python -c "import Cython; print(Cython.__version__)"
python -c "import numpy; print(numpy.get_include())"

# Compile Cython code manually
C_INCLUDE_PATH=$(python -c 'import numpy; print(numpy.get_include())') cythonize -a -i cython_sdft_functions.pyx

# Install project and dependencies
pip install -e .

# Run Python script
python real_time_audio_visualization.py

# Deactivate virtual environment
deactivate

