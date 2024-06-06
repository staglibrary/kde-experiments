"""
This module provides functions for interacting with different datasets.
"""
from typing import Optional, List
import keras.datasets.mnist
import numpy as np
import scipy.io
import skimage.transform
import skimage.measure
import skimage.filters
import sklearn.metrics
from sklearn import datasets
from sklearn import random_projection
import random
import math
import matplotlib
import matplotlib.pyplot as plt
import os
from matplotlib import image


class Dataset(object):
    """
    This base class represents a dataset, for clustering. A dataset might consist of some combination of:
      - raw numerical data
      - a ground truth clustering
    """

    def __init__(self, raw_data=None):
        """
        Intiialise the dataset, optionally specifying a data file.
        """
        self.raw_data = raw_data
        self.gt_labels: Optional[List[int]] = None
        self.load_data()
        self.num_data_points = self.raw_data.shape[0] if self.raw_data is not None else 0
        self.data_dimension = self.raw_data.shape[1] if self.raw_data is not None else 0

    def load_data(self):
        """
        Load the data for this dataset. The implementation may differ significantly
        from dataset to dataset.
        """
        pass

    def plot(self, labels):
        """
        Plot the dataset with the given labels.
        """
        if self.data_dimension > 3:
            print("Cannot plot data with dimensionality above 3.")
            return

        if self.data_dimension == 2:
            plt.scatter(self.raw_data[:, 0], self.raw_data[:, 1], c=labels, marker='.')
        else:
            # Create the figure
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(self.raw_data[:, 0], self.raw_data[:, 1], self.raw_data[:, 2], c=labels, marker='.')
        plt.show()

    def ari(self, labels):
        """
        Compute the Adjusted Rand Index of the given candidate labels.
        """
        if self.gt_labels is not None:
            return sklearn.metrics.adjusted_rand_score(self.gt_labels, labels)
        else:
            return 0

    def normalise(self):
        """
        Normalise the data to lie between 0 and 1.
        """
        # Get the maximum and minimum values in each dimension
        self.data_dimension = self.raw_data.shape[1] if self.raw_data is not None else 0
        dimension_max_vals = [float('-inf')] * self.data_dimension
        dimension_min_vals = [float('inf')] * self.data_dimension
        for d in range(self.data_dimension):
            for point in self.raw_data:
                if point[d] > dimension_max_vals[d]:
                    dimension_max_vals[d] = point[d]
                if point[d] < dimension_min_vals[d]:
                    dimension_min_vals[d] = point[d]

        # Find the amount we need to scale down by
        scale_factor = 0
        for d in range(self.data_dimension):
            this_range = dimension_max_vals[d] - dimension_min_vals[d]
            if this_range > scale_factor:
                scale_factor = this_range

        # Normalise all of the data
        for i, point in enumerate(self.raw_data):
            for d in range(self.data_dimension):
                self.raw_data[i, d] = (point[d] - dimension_min_vals[d]) / scale_factor

    def __repr__(self):
        return self.__str__()


class TwoMoonsDataset(Dataset):

    def __init__(self, n=1000):
        """
        Create an instance of the two moons dataset with the specified number of
        data points.
        """
        self.n = n
        super(TwoMoonsDataset, self).__init__()

    def load_data(self):
        self.raw_data, self.gt_labels = datasets.make_moons(n_samples=self.n, noise=0.05)
        self.data_dimension = 2
        self.normalise()

    def __str__(self):
        return f"twoMoonsDataset(n={self.num_data_points})"


class BlobsDataset(Dataset):

    def __init__(self, n=1000, k=10, d=3, std=1):
        """
        Create an instance of the blobs dataset.
        """
        self.n = n
        self.k = k
        self.d = d
        self.std = std
        super(BlobsDataset, self).__init__()

    def load_data(self):
        self.raw_data, self.gt_labels = datasets.make_blobs(
            n_samples=self.n, n_features=self.d, centers=self.k, cluster_std=self.std)
        self.data_dimension = self.d
        self.normalise()

    def __str__(self):
        return f"blobsDataset(n={self.n}, k={self.k}, d={self.d})"

