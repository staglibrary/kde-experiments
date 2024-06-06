#ifndef KDE_EXPERIMENTS_DATA_H
#define KDE_EXPERIMENTS_DATA_H

#define H5_USE_EIGEN
#include <highfive/H5Easy.hpp>

// STAG definitions
#include "definitions.h"

/**
 * Load a dataset from an HDF5 file.
 *
 * The dataset must be a simple matrix of numeric data.
 *
 * @param filename the name of the file from which to load the dataset
 * @param dataset the name of the dataset to be loaded
 * @return an eigen matrix containing the dataset
 * @throws HighFive::FileException if the specified file is not found
 * @throws Highfive::DatasetException if the specified dataset is not found
 */
DenseMat load_hdf5(const std::string& filename, const std::string& dataset);

/**
 * List the dataset in the given HDF5 file.
 *
 * @param filename the name of the file from which to list the datasets.
 * @return a vector of the dataset names
 * @throws HighFive::FileException if the specified file is not found
 */
std::vector<std::string> list_datasets_hdf5(const std::string& filename);

StagReal get_bandwidth(const std::string& filename, const std::string& attribute_name);

#endif //KDE_EXPERIMENTS_DATA_H
