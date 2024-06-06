"""Analyse the results of the KDE experiments."""
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

TARGET_RELATIVE_ERROR = 0.1

DEANN_ALGORITHMS = ['ann-faiss', 'ann-permuted-faiss', 'naive', 'rs', 'rsp', 'sklearn-balltree']
LATEX_ALG_NAMES = {'stag': 'stag',
                   'ann-faiss': 'DEANN',
                   'ann-permuted-faiss': 'DEANNP',
                   'naive': 'naive',
                   'rs': 'rs',
                   'rsp': 'rsp',
                   'sklearn-balltree': 'sklearn'
                   }
ALG_PLOT_COLORS = {'stag': 'red',
                   'ann-faiss': 'skyblue',
                   'ann-permuted-faiss': 'blue',
                   'naive': 'black',
                   'rs': 'orange',
                   'rsp': 'green',
                   'sklearn-balltree': 'brown'
                   }
dataset_mus = {
    "aloi": ["01"],
    "covtype": ["01"],
    "glove": ["01"],
    "mnist": ["01"],
    "msd": ["01"],
    "shuttle": ["01"],
}

dataset_ns = {'shuttle': 38000,
              'aloi': 107000,
              'msd': 515345,
              'glove': 1193514,
              'mnist': 70000,
              'covtype': 581012}

dataset_ds = {'shuttle': 9,
              'aloi': 128,
              'msd': 90,
              'glove': 100,
              'mnist': 728,
              'covtype': 54}


def nice_plot(xs, ys, x_label, y_label, filename=None):
    """
    Create a standard plotting function to create consistent looking plots.
    """
    # Plotting the function
    plt.figure(figsize=(4, 3))
    plt.plot(xs, ys, color='skyblue', linewidth=5)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)

    if filename is not None:
        plt.savefig(filename, format="pdf", bbox_inches="tight")

    plt.show()


def get_stag_filename(dataset, mu):
    return f"../stag/results/{dataset}.{mu}.csv"


def get_deann_filename(dataset, mu):
    return f"../DEANN/results/{dataset}.{mu}.csv"


def load_stag_results(filename):
    try:
        stag_data = pd.read_csv(filename, skipinitialspace=True)
        return stag_data
    except FileNotFoundError as e:
        return None


def load_deann_results(filename):
    try:
        deann_data = pd.read_csv(filename, skipinitialspace=True)
        return deann_data
    except FileNotFoundError as e:
        return None


def get_best_running_times(dataset, mu, err):
    stag_df = load_stag_results(get_stag_filename(dataset, mu))
    deann_df = load_deann_results(get_deann_filename(dataset, mu))

    # Create a dictionary of algorithm names to running times
    best_times = {}

    # Find the stag result with the best running time
    if stag_df is not None:
        stag_good = stag_df[stag_df['rel_err'] <= err]
        best_query_time = stag_good.query_time.min()
        best_times['stag'] = best_query_time
    else:
        best_times['stag'] = -1

    # Find the DEANN results with the best running times
    if deann_df is not None:
        deann_good = deann_df[deann_df['rel_err'] <= err]
        for alg_name in DEANN_ALGORITHMS:
            alg_df = deann_good[deann_good['algorithm'] == alg_name]
            best_query_time = alg_df.query_time.min()
            best_deann_result = alg_df[alg_df['query_time'] == best_query_time]
            if len(best_deann_result) > 0:
                best_times[alg_name] = best_query_time
            else:
                best_times[alg_name] = -1
    else:
        for alg_name in DEANN_ALGORITHMS:
            best_times[alg_name] = -1

    return best_times


def create_best_times_table():
    """Create a latex table with the best running times.."""
    table_filename = "../figures/times_table.tex"
    table_preamble = """\\begin{table} [htb]
    \\caption{The best per-query update times achieving a relative error of at most $0.1$.}
    \\label{tab:kde_times}
    \\centering
    \\begin{tabular}"""

    table_content = ""
    table_format_str = None
    table_header = None
    for dataset in dataset_mus.keys():
        for mu in dataset_mus[dataset]:
            best_times = get_best_running_times(dataset, mu, TARGET_RELATIVE_ERROR)
            table_content += "    " + dataset + " & " + " & ".join([f"{v: .3f}" if v > 0 else "-" for v in best_times.values()]) + " \\\\\n"

            if table_format_str is None:
                table_format_str = "{cc" + "".join(["c" for x in best_times.keys()]) + "}\n    \\toprule\n"
            if table_header is None:
                table_header = "    & \multicolumn{7}{c}{Algorithm} \\\\ \n"
                table_header += "    \cmidrule(l){2-8} \n"
                table_header += "    dataset & " + " & ".join(
                    [LATEX_ALG_NAMES[alg] for alg in best_times.keys()]) + "\\\\ \n    \\midrule\n"

    table_endamble = """    \\bottomrule
  \\end{tabular}
\\end{table}"""

    table_str = table_preamble + table_format_str + table_header + table_content + table_endamble

    with open(table_filename, 'w') as fout:
        fout.write(table_str)


def plot_running_times(dataset, mu, ymax=4):
    errors = []
    all_times = []
    for e in np.arange(0.01, 0.2, 0.01):
        errors.append(e)
        best_times = get_best_running_times(dataset, mu, e)
        all_times.append(best_times)

    # Plotting the function
    plt.figure(figsize=(4, 3))
    plt.xlabel("Error")
    plt.ylabel("Query Time")
    plt.grid(True)

    for alg in LATEX_ALG_NAMES.keys():
        include_this_alg = False
        times = []
        for d in all_times:
            if d[alg] == -1:
                times.append(float('nan'))
            else:
                if d[alg] < ymax:
                    include_this_alg = True
                times.append(d[alg])
        if include_this_alg:
            plt.plot(errors, times, label=LATEX_ALG_NAMES[alg], linewidth=2, color=ALG_PLOT_COLORS[alg])

    if dataset == 'aloi':
        plt.legend(loc="upper right")
    ax = plt.gca()
    ax.set_ylim([0, ymax])
    plt.savefig(f"../figures/times_{dataset}_{mu}.pdf", format="pdf", bbox_inches="tight")
    plt.show()


def plot_all_running_times():
    ymaxs = {'aloi': {'01': 4, '001': 1.5},
             'glove': {'01': 0.1},
             'shuttle': {'01': 0.5},
             'msd': {'01': 0.5},
             'mnist': {'01': 0.5},
             'covtype': {'01': 0.2}}
    for dataset in ymaxs.keys():
        for mu in ymaxs[dataset].keys():
            plot_running_times(dataset, mu, ymax=ymaxs[dataset][mu])


def main():
    create_best_times_table()
    plot_all_running_times()


if __name__ == "__main__":
    main()
