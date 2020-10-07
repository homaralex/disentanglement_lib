import argparse

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scripts.load_aggregated_results import DATASET_COL_STR, MODEL_COL_STR, RESULTS_DIR, DATASETS, DIS_METRICS, \
    get_dataset_df, get_metric_df, get_method_df, HUMAN_READABLE_NAMES, PLOT_DIR, get_reg_col_name

_BASELINE_CACHE_FILE = RESULTS_DIR / 'dlib_baseline_cache.json'
_OUT_DIR = PLOT_DIR / 'greedy'
_OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_baseline():
    # TODO load annealed for shapes3D
    df_shapes = pd.read_json(RESULTS_DIR / 'test_results_cpu02.json')
    df_shapes = df_shapes.loc[
        (df_shapes[DATASET_COL_STR].str.contains('shapes3d'))
        & (df_shapes[MODEL_COL_STR].str.contains('beta_vae'))
        ]

    if _BASELINE_CACHE_FILE.exists():
        df_dlib = pd.read_json(_BASELINE_CACHE_FILE)
    else:
        df_dlib = pd.read_json(RESULTS_DIR / 'dlib_results.json')
        df_dlib = df_dlib.loc[
            (df_dlib[MODEL_COL_STR].str.contains('beta_vae'))
            | (df_dlib[MODEL_COL_STR].str.contains('annealed_vae'))
            ]
        df_dlib.to_json(_BASELINE_CACHE_FILE)

    df_baseline = pd.concat((df_shapes, df_dlib), sort=True)

    return df_baseline


def load_results(result_files):
    return pd.concat((pd.read_json(RESULTS_DIR / results_file) for results_file in result_files), sort=True)


def plot_fig_15(df):
    fig, axes = plt.subplots(nrows=len(DATASETS), ncols=len(DIS_METRICS), figsize=(30, 20))

    for row_idx, dataset in enumerate(DATASETS):
        df_dataset = get_dataset_df(df, dataset)

        for col_idx, metric in enumerate(DIS_METRICS):
            ax = axes[row_idx, col_idx]
            df_metric, metric_col_name = get_metric_df(df_dataset, metric)
            x_ranges = []

            methods = df[MODEL_COL_STR].unique()
            for method in methods:
                reg_col_name = get_reg_col_name(method)
                df_method = get_method_df(df_metric, method)

                grouped_df = df_method.groupby(reg_col_name)[metric_col_name].mean()

                x_range = list(range(len(grouped_df)))
                x_ranges.append(len(x_range))
                sns.lineplot(
                    x=x_range,
                    y=grouped_df.values,
                    ax=ax,
                    # TODO change to readable
                    label=method,
                    # label=HUMAN_READABLE_NAMES[method],
                    linewidth=4,
                )

            ax.get_legend().remove()
            ax.grid()
            ax_kwargs = {}
            if row_idx == 0:
                ax_kwargs['title'] = HUMAN_READABLE_NAMES[metric]
            if col_idx == 0:
                ax_kwargs['ylabel'] = 'Value'
            if row_idx == len(DATASETS) - 1:
                ax_kwargs['xlabel'] = 'Regularization strength'
            if col_idx == len(DIS_METRICS) - 1:
                ax.yaxis.set_label_position('right')
                ax_kwargs['ylabel'] = HUMAN_READABLE_NAMES[dataset]
            ax.set(**ax_kwargs)

    axes[0, 2].legend(loc='upper center', bbox_to_anchor=(1.5, 1.6), ncol=len(methods) + 2)

    plt.savefig(PLOT_DIR / 'greedy_fig_15.png')
    plt.show()
    plt.close(fig)


def plot_violin(df):
    for dataset in DATASETS:
        print(dataset)

        df_dataset = get_dataset_df(df, dataset)
        fig, axes = plt.subplots(nrows=len(DIS_METRICS), ncols=1, figsize=(30, 30))

        for row_idx, dis_metric in enumerate(DIS_METRICS):
            df_metric, metric_col_name = get_metric_df(df_dataset, dis_metric)
            ax = axes[row_idx]

            sns.violinplot(
                x='train_config.model.name',
                y=metric_col_name,
                data=df_metric,
                cut=0,
                ax=ax,
                hue='model_type',
            )
            ax.set_ylabel(HUMAN_READABLE_NAMES[dis_metric])
            for tick in ax.get_xticklabels():
                tick.set_rotation(25)

        fig.suptitle(dataset)
        fig.savefig(_OUT_DIR / f'{dataset}_violin.png')
        plt.close(fig)


def main(result_files):
    df_baseline = load_baseline()
    df_results = load_results(result_files)

    df = pd.concat((df_baseline, df_results), sort=True)

    plot_violin(df)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('result_files', type=str, nargs='+', )
    args = parser.parse_args()

    main(args.result_files)
