# KDE Experiments
This directory contains code for running experiments with
KDE implementations in high dimensions.
The stag directory contains the code for running experiments with the stag KDE
implementation.
the DEANN directory contains the code for running experiments with the
alternative KDE algorithms. The code in this directory is derived from
the DEANN experiments code available
[here](https://github.com/mkarppa/deann-experiments).

## Installing C++ dependencies
The stag experiments rely on the following C++ libraries:
- STAG
- HDF5
- HighFive

STAG can be installed by following the
[installation instructions](https://staglibrary.io/docs/cpp/docs-2.0.0/getting-started.html)
on the STAG website.

The HDF5 library is available
[here](https://portal.hdfgroup.org/downloads/hdf5/hdf5_1_14_3.html).
Follow the installation the instructions in the official documentation.
The following commands *might* work on a linux system.
```
mkdir temp
wget https://hdf-wordpress-1.s3.amazonaws.com/wp-content/uploads/manual/HDF5/HDF5_1_14_3/src/hdf5-1.14.3.tar.gz
gunzip < hdf5-1.14.3.tar.gz | tar xf -
cd hdf5-1.14.3
./configure --prefix=/usr/local/hdf5 --enable-cxx
make
make check
make install
make check-install 
```

Finally, install the high five module from github.

```
git clone --recursive https://github.com/BlueBrain/HighFive.git
cd HighFive
git checkout v2.8.0
cmake -DCMAKE_INSTALL_PREFIX=/usr/lib/ -DHIGHFIVE_USE_BOOST=Off -DCMAKE_PREFIX_PATH=/usr/local/hdf5/ -B build .
cmake --build build --parallel
cmake --install build
```

Note that if you install the dependencies to a different location, you need to run cmake with
a different path for '-DCMAKE_INSTALL_PREFIX'.

## Compiling the experiments code
Then, you can compile the experiments code with the following commands in the
`stag` directory.

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target kde_experiments
```

## Installing Python dependencies
The easiest method to run the experiments is to create a conda environment 
the following commands.

```bash
conda create kde-experiments
conda activate kde-experiments
```

Then, install all python requirements with the following.

```bash
python -m pip install requirements.txt
```

## Downloading the test data
The tools directory provides the `preprocess_datasets` python script for downloading the data.

From the tools directory, run this script, passing the name of a dataset as the `--dataset` parameter.
For example:

```python preprocess_datasets --dataset aloi```

This will download the data into the 'data' directory, and add an HDF5 file containing:

- train, test, and validation splits of the original dataset
- the ground truth KDE values for the gaussian kernel for different bandwidths

The bandwidths used are stored as attributes on the HDF5 file: for each set of ground truth answers,
the average KDE value, and the bandwidth used are stored as a tuple.

You can use a tool such as [this](https://myhdf5.hdfgroup.org/) to visualise the HDF5 file in order to understand its
structure.

## Running the STAG KDE experiments
Once the dataset has been downloaded, and the STAG experiment code has been compiled,
from the build directory, run the following command.

```bash
./kde_experiments ../../data/<dataset>.hdf5 ../results/<dataset>.01.csv 0.01
```

This will run many KDE experiments with the STAG implementation on the provided
dataset and write the results to the `../results/<dataset>.01.csv` file.

## Running other experiments
The DEANN directory contains the code for running the experiments with other 
KDE algorithms.
The code has been only slightly modified from
[the original](https://github.com/mkarppa/deann-experiments)
by the DEANN authors.

To run the experiments, from the `DEANN` directory, run

```bash
python run_exp.py --dataset <dataset> --no-docker --kernel "gaussian" --kde-value 0.01
```

Then, to generate the results file, run

```bash
python data_export.py --dataset <dataset> --mu 0.01 -o results/<dataset>.01.csv
```

## Analysing the results
Once the experiments have been run, to generate the results table and figures,
we provide the `analyse_results.py` script in the `tools` directory.

Running this script with 
```bash
python analyse_results.py
```
will produce plots and a latex table in the figures directory containing
the running time comparison of the algorithms.
