"""
Run some spectral clustering experiments.
"""
import os
from time import time
import matplotlib.ticker
import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse
import clusteralgs
import datasets

import warnings
warnings.filterwarnings("ignore")


SRC_DIR = os.getcwd()
PARENT_DIR = os.path.abspath(os.path.join(SRC_DIR, os.pardir))

ALGORITHMS_TO_COMPARE = ["Scipy RBF", "Scipy kNN", "FAISS HNSW", "FAISS IVF", "IFGT FSC", "STAG ASG"]

linestyle_map = {'Scipy KNN': 'dotted',
                 'Scipy RBF': 'dashed',
                 'FAISS HNSW': 'dashed',
                 'FAISS IVF': 'dotted',
                 'IFGT FSC': 'dashed',
                 'STAG ASG': 'solid',
                 }

ALG_PLOT_COLORS = {
    'Scipy RBF': 'black',
    'Scipy KNN': 'green',
    'FAISS HNSW': 'skyblue',
    'FAISS IVF': 'blue',
    'IFGT FSC': 'orange',
    'STAG ASG': 'red',
}

LATEX_ALG_NAMES = {
    'Scipy RBF': 'Sklearn FC',
    'Scipy KNN': 'Sklearn kNN',
    'FAISS HNSW': 'FAISS HNSW',
    'FAISS IVF': 'FAISS IVF',
    'IFGT FSC': 'MS',
    'STAG ASG': 'stag',
}

class ExperimentRunData(object):

    def __init__(self, dataset, running_time, labels, extra_info=None):
        self.running_time = running_time
        self.ari = dataset.ari(labels)
        self.extra_info = extra_info
        self.num_data_points = dataset.num_data_points
        self.dataset = dataset
        self.labels = labels


def blobs_dimension_experiment():
    print("Clustering blobs dataset.")
    k = 10
    n = 10000

    algorithms_to_compare = {
        "Scipy RBF": (lambda ds: clusteralgs.rbf_spectralcluster(ds, k, gamma=200)),
        "Scipy KNN": (lambda ds: clusteralgs.knn_spectralcluster(ds, k)),
        "FAISS HNSW": (lambda ds: clusteralgs.faiss_hnsw_spectral_cluster(ds, k)),
        "FAISS IVF": (lambda ds: clusteralgs.faiss_ivf_spectral_cluster(ds, k)),
        "IFGT FSC": (lambda ds: clusteralgs.fast_spectral_cluster_ifgt(ds, k, 0.1)),
        "STAG ASG": (lambda ds: clusteralgs.stag_asg_spectral_cluster(ds, k, a=200)),
    }

    max_data_size = {
        "Scipy RBF": float('inf'),
        "Scipy KNN": float('inf'),
        "FAISS HNSW": float('inf'),
        "FAISS IVF": float('inf'),
        "IFGT FSC": float('inf'),
        "STAG ASG": float('inf'),
    }

    max_time_cutoff = {
        "Scipy RBF": 60,
        "Scipy KNN": 60,
        "FAISS HNSW": 60,
        "FAISS IVF": 60,
        "IFGT FSC": 60,
        "STAG ASG": 60,
    }

    cut_off = {k: False for k in max_time_cutoff}

    experimental_data = {k: [] for k in algorithms_to_compare}
    for da in np.linspace(2, 10, num=8):
        d = int(da)

        print(f"Number of dimensions = {d}")
        dataset = datasets.BlobsDataset(n=n, k=k, d=d)

        for alg_name, func in algorithms_to_compare.items():
            if not cut_off[alg_name] and n <= max_data_size[alg_name]:
                tstart = time()
                sc_labels = func(dataset)
                tend = time()
                duration = tend - tstart
                experimental_data[alg_name].append(ExperimentRunData(dataset, duration, sc_labels))
                print(f"{alg_name}: {duration:0.3f} seconds, {experimental_data[alg_name][-1].ari:0.3f} ARI")

                if duration > max_time_cutoff[alg_name]:
                    cut_off[alg_name] = True
            else:
                print(f"Skipping {alg_name}...")

        print("")

    # Save the results
    with open(os.path.join(PARENT_DIR, "results/blobs/dim_results.pickle"), 'wb') as fout:
        pickle.dump(experimental_data, fout)


def blobs_n_experiment():
    print("Clustering blobs dataset.")
    k = 10
    d = 100

    algorithms_to_compare = {
        "Scipy RBF": (lambda ds: clusteralgs.rbf_spectralcluster(ds, k, gamma=20)),
        "Scipy KNN": (lambda ds: clusteralgs.knn_spectralcluster(ds, k)),
        "FAISS HNSW": (lambda ds: clusteralgs.faiss_hnsw_spectral_cluster(ds, k)),
        "FAISS IVF": (lambda ds: clusteralgs.faiss_ivf_spectral_cluster(ds, k)),
        "STAG ASG": (lambda ds: clusteralgs.stag_asg_spectral_cluster(ds, k, a=1)),
    }

    max_data_size = {
        "Scipy RBF": 20000,
        "Scipy KNN": float('inf'),
        "FAISS HNSW": float('inf'),
        "FAISS IVF": float('inf'),
        "STAG ASG": float('inf'),
    }

    max_time_cutoff = {
        "Scipy RBF": 300,
        "Scipy KNN": 300,
        "FAISS HNSW": 300,
        "FAISS IVF": 300,
        "STAG ASG": 300,
    }

    cut_off = {k: False for k in max_time_cutoff}

    experimental_data = {k: [] for k in algorithms_to_compare}
    for na in np.logspace(3, 5, num=30):
        n = int(na)

        print(f"Number of points = {n}")
        dataset = datasets.BlobsDataset(n=n, k=k, d=d)

        for alg_name, func in algorithms_to_compare.items():
            if not cut_off[alg_name] and n <= max_data_size[alg_name]:
                tstart = time()
                sc_labels = func(dataset)
                tend = time()
                duration = tend - tstart
                experimental_data[alg_name].append(ExperimentRunData(dataset, duration, sc_labels))
                print(f"{alg_name}: {duration:0.3f} seconds, {experimental_data[alg_name][-1].ari:0.3f} ARI")

                if duration > max_time_cutoff[alg_name]:
                    cut_off[alg_name] = True
            else:
                print(f"Skipping {alg_name}...")

        print("")

    # Save the results
    with open(os.path.join(PARENT_DIR, "results/blobs/n_results.pickle"), 'wb') as fout:
        pickle.dump(experimental_data, fout)


def create_plot(filename, all_data, x_label, y_label, x_lambda, y_lambda,
                xmin=None, xmax=None, ymin=None, ymax=None, excluded_algs=None,
                show_legend=True):
    if excluded_algs is None:
        excluded_algs = []

    plt.figure(figsize=(4, 3))
    plt.grid(True)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    for alg in LATEX_ALG_NAMES.keys():
        if alg not in excluded_algs:
            data = all_data[alg]

            plt.plot([x_lambda(d) for d in data],
                     [y_lambda(d) for d in data],
                     label=LATEX_ALG_NAMES[alg],
                     linewidth=2,
                     color=ALG_PLOT_COLORS[alg])

    if show_legend:
        plt.legend(loc='best')
    ax = plt.gca()
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    if ymin is not None:
        ax.set_ylim(ymin, ymax)
    if xmin is not None:
        ax.set_xlim(xmin, xmax)
    plt.savefig(filename, format="pdf", bbox_inches="tight")
    plt.show()


def blobs_experiment_plot():
    with open(os.path.join(PARENT_DIR, "results/blobs/dim_results.pickle"), 'rb') as fin:
        experimental_data = pickle.load(fin)

    create_plot("blobs_dim.pdf", experimental_data, "Number of dimensions",
                "Running time (s)", (lambda d: d.dataset.data_dimension),
                (lambda d: d.running_time), ymin=0, ymax=60, xmin=2, xmax=10)

    with open(os.path.join(PARENT_DIR, "results/blobs/n_results.pickle"), 'rb') as fin:
        experimental_data = pickle.load(fin)

    create_plot("blobs_n.pdf", experimental_data, "Number of data points",
                "Running time (s)", (lambda d: d.num_data_points),
                (lambda d: d.running_time), excluded_algs=["IFGT FSC", "Scipy RBF"],
                ymin=0, ymax=300, show_legend=False)

    create_plot("blobs_n_small.pdf", experimental_data, "Number of data points",
                "Running time (s)", (lambda d: d.num_data_points),
                (lambda d: d.running_time), excluded_algs=["IFGT FSC"],
                ymin=0, ymax=50,
                xmin=0, xmax=20000)


def two_moons_experiment():
    print("Clustering two moons dataset.")
    k = 2

    algorithms_to_compare = {
        "Scipy RBF": (lambda ds: clusteralgs.rbf_spectralcluster(ds, k, gamma=200)),
        "Scipy KNN": (lambda ds: clusteralgs.knn_spectralcluster(ds, k)),
        "FAISS HNSW": (lambda ds: clusteralgs.faiss_hnsw_spectral_cluster(ds, k)),
        "FAISS IVF": (lambda ds: clusteralgs.faiss_ivf_spectral_cluster(ds, k)),
        "IFGT FSC": (lambda ds: clusteralgs.fast_spectral_cluster_ifgt(ds, k, 0.1)),
        "STAG ASG": (lambda ds: clusteralgs.stag_asg_spectral_cluster(ds, k, a=200)),
    }

    max_data_size = {
        "Scipy RBF": float('inf'),
        "Scipy KNN": float('inf'),
        "FAISS HNSW": float('inf'),
        "FAISS IVF": float('inf'),
        "IFGT FSC": float('inf'),
        "STAG ASG": float('inf'),
    }

    max_time_cutoff = {
        "Scipy RBF": 60,
        "Scipy KNN": 60,
        "FAISS HNSW": 60,
        "FAISS IVF": 60,
        "IFGT FSC": 60,
        "STAG ASG": 60,
    }

    cut_off = {k: False for k in max_time_cutoff}

    experimental_data = {k: [] for k in algorithms_to_compare}
    for na in np.logspace(3, 6, num=30):
        n = int(na)

        print(f"Number of data points = {n}")
        dataset = datasets.TwoMoonsDataset(n=n)

        for alg_name, func in algorithms_to_compare.items():
            if not cut_off[alg_name] and n <= max_data_size[alg_name]:
                tstart = time()
                sc_labels = func(dataset)
                tend = time()
                duration = tend - tstart
                experimental_data[alg_name].append(ExperimentRunData(dataset, duration, sc_labels))
                print(f"{alg_name}: {duration:0.3f} seconds, {experimental_data[alg_name][-1].ari:0.3f} ARI")

                if duration > max_time_cutoff[alg_name]:
                    cut_off[alg_name] = True
            else:
                print(f"Skipping {alg_name}...")

        print("")

    # Save the results
    with open(os.path.join(PARENT_DIR, "results/twomoons/results.pickle"), 'wb') as fout:
        pickle.dump(experimental_data, fout)


def two_moons_experiment_plot():
    with open(os.path.join(PARENT_DIR, "results/twomoons/results.pickle"), 'rb') as fin:
        experimental_data = pickle.load(fin)

    create_plot("moons.pdf", experimental_data, "Number of data points",
                "Running time (s)", (lambda d: d.num_data_points),
                (lambda d: d.running_time), ymin=0, ymax=80,
                xmin=0, xmax=100000, show_legend=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Run experiments.")
    parser.add_argument('command', type=str, choices=['plot', 'run'])
    parser.add_argument('experiment', type=str,
                        choices=['moons', 'blobs'])
    return parser.parse_args()


def main():
    args = parse_args()

    ########################
    # Two Moons
    ########################
    if args.experiment == 'moons':
        if args.command == 'run':
            two_moons_experiment()
        elif args.command == 'plot':
            two_moons_experiment_plot()

    ########################
    # Blobs
    ########################
    if args.experiment == 'blobs':
        if args.command == 'run':
            blobs_dimension_experiment()
            blobs_n_experiment()
        elif args.command == 'plot':
            blobs_experiment_plot()


if __name__ == "__main__":
    main()
