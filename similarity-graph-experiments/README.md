# Similarity Graph Experiments
This directory contains code for performing experiments on spectral clustering with different similarity graphs.

## Build Instructions
One of the algorithms we compare against is the fast approximate similarity graph constructed
using the Fast Gauss Transform from [this repository](https://github.com/pmacg/kde-similarity-graph).
This algorithm is written in C++, in the `src/cpp/` directory, with a python wrapper around this C++ code.
To compile the code, follow the instructions below.
It is recommended to use a Python conda environment, as this is the easiest way to install the project dependencies.

### Install the C++ dependencies

The C++ code requires the following libraries to be installed.
- Eigen
- Spectra

You should refer to their documentation for installation instructions, although
the following should work on a standard linux system.

```bash
# Create a directory to work in
mkdir libraries
cd libraries

# Install Eigen
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
tar xzvf eigen-3.4.0.tar.gz
cd eigen-3.4.0
mkdir build_dir
cd build_dir
cmake ..
sudo make install
cd ../..

# Install Spectra
wget https://github.com/yixuan/spectra/archive/v1.0.1.tar.gz
tar xzvf v1.0.1.tar.gz
cd spectra-1.0.1
mkdir build_dir
cd build_dir
cmake ..
sudo make install
cd ../..
```

### Compile the C++ Python extension

In the directory containing this README file, run the following
commands:

```bash
conda install --file requirements.txt
python setup.py build_ext --inplace
```

This will compile the C++ code and create an extension file which can be imported
by the Python code.

### Install the FAISS library
We compare our algorithm against the approximate nearest neighbour graphs constructed
with the FAISS library. Install FAISS with conda.

```bash
conda install -c pytorch faiss-cpu
```

## Running the experiments

There are two experiments included in this directory.

### Two moons experiment

To run the experiment for comparing the algorithms' running time on the two
moons dataset, run the following conda command.

```bash
conda run --no-capture-output python main.py run moons
```

Then, to create a figure displaying the results, run the following.

```bash
conda run --no-capture-output python main.py plot moons
```

### Blobs experiment

To run the experiments with the blobs dataset, run the following command.

```bash
conda run --no-capture-output python main.py run blobs
```

Then, to create a figure displaying the results, run the following.

```bash
conda run --no-capture-output python main.py plot blobs
```

