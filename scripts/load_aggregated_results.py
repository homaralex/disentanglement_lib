import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import disentanglement_lib.utils.hyperparams as h

MODEL_COL_STR = 'train_config.model.name'
DATASET_COL_STR = 'train_config.dataset.name'
RESULTS_DIR = Path('aggregated_results')
DLIB_RESULTS_PATH = RESULTS_DIR / 'dlib_beta_vae_results.json'

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

DIS_METRICS = (
    'beta_vae_sklearn',
    'factor_vae_metric',
    'evaluation_results.discrete_mig',
    'evaluation_results.modularity_score',
    'evaluation_results.SAP_score',
    'evaluation_results.disentanglement',

    # 'train_results.loss',
    # 'train_results.regularizer',

    # 'evaluation_results.completeness',
    # 'evaluation_results.informativeness_test',
    # 'evaluation_results.explicitness_score_test',

    # 'evaluation_results.mutual_info_score',
    # 'evaluation_results.gaussian_total_correlation',
    # 'evaluation_results.gaussian_wasserstein_correlation_norm',
)
UNSUP_METRICS = (
    'train_results.reconstruction_loss',
    'train_results.kl_loss',
    'train_results.elbo',
)
DATASETS = (
    'dsprites_full',
    'scream_dsprites',
    'shapes3d',
    'cars3d',
)
METHODS = (
    'masked',
    'dim_wise_mask_l1_col',
    'dim_wise_mask_l1',
    # 'dim_wise_mask_l1_row',
    # 'weight_decay',
)

PLOT_DIR = Path('plots')
PLOT_DIR.mkdir(exist_ok=True)


def load_results():
    return pd.concat((pd.read_json(RESULTS_DIR / results_file) for results_file in args.result_files), sort=True)


def get_dataset_df(df, dataset):
    return df.loc[df[DATASET_COL_STR] == f"'{dataset}'"]


def get_metric_df(df, metric):
    if metric in ('beta_vae_sklearn', 'factor_vae_metric'):
        metric_df = df.loc[df['evaluation_config.evaluation.name'].str.contains(metric)]
        metric = 'evaluation_results.eval_accuracy'
    else:
        metric_df = df.loc[df[metric].notna()]

    return metric_df, metric


def get_method_df(df, method):
    return df.loc[df[MODEL_COL_STR].str.contains(method)]


def get_reg_col_name(method):
    if 'masked' in method:
        return 'train_config.conv_encoder.perc_sparse'
    elif 'dim_wise' in method:
        return 'train_config.dim_wise_l1_vae.lmbd_l1'


def plot_results(
        df,
        reg_weight_col,
        out_dir,
        dataset,
):
    fig_violin, axes_violin = plt.subplots(nrows=3, ncols=4, figsize=(30, 30))
    fig_box, axes_box = plt.subplots(nrows=3, ncols=4, figsize=(30, 30))
    fig_mean, axes_mean = plt.subplots(nrows=3, ncols=4, figsize=(30, 30))
    for metric, ax_violin, ax_box, ax_mean in zip(
            DIS_METRICS,
            axes_violin.flatten(),
            axes_box.flatten(),
            axes_mean.flatten(),
    ):
        metric_df, metric = get_metric_df(df, metric)
        print()
        print(metric_df.groupby(MODEL_COL_STR)[metric].mean().reset_index().sort_values(metric, ascending=False))

        metric_df = metric_df.sort_values(reg_weight_col)

        sns.violinplot(
            x=MODEL_COL_STR,
            y=metric,
            data=metric_df,
            cut=0,
            ax=ax_violin,
        )
        for tick in ax_violin.get_xticklabels():
            tick.set_rotation(45)

        sns.boxplot(
            x=MODEL_COL_STR,
            y=metric,
            data=metric_df,
            ax=ax_box,
        )
        for tick in ax_box.get_xticklabels():
            tick.set_rotation(45)

        # group and aggregate to obtain means per model
        metric_df = metric_df.groupby(reg_weight_col)[metric].mean()
        sns.stripplot(
            x=list(map("{:.2E}".format, metric_df.index.values)),
            y=metric_df.values,
            ax=ax_mean,
            size=25,
        )
        ax_mean.set_ylabel(metric)

        for tick in ax_mean.get_xticklabels():
            tick.set_rotation(45)

    fig_violin.savefig(out_dir / f'{dataset}_violin.png')
    fig_box.savefig(out_dir / f'{dataset}_box.png')
    fig_mean.savefig(out_dir / f'{dataset}_mean.png')

    for fig in (fig_violin, fig_box, fig_mean):
        plt.close(fig)


def load_dlib_df():
    return pd.read_json(DLIB_RESULTS_PATH)


def print_rankings(df):
    ranking_dict = {dataset: defaultdict(list) for dataset in (DATASETS + ('all',))}
    for col_name in (
            MODEL_COL_STR,
            # 'model_idx',
    ):
        for dataset in DATASETS:
            print()
            print(dataset)
            dset_df = get_dataset_df(df, dataset)

            for metric in DIS_METRICS:
                metric_df, metric = get_metric_df(dset_df, metric)
                print()
                ranking = metric_df.groupby(col_name)[metric].mean().reset_index().sort_values(metric, ascending=False)
                print(ranking[:5])

                for idx, row in enumerate(ranking.values):
                    model_name = row[0]
                    ranking_dict[dataset][model_name].append(idx)
                    ranking_dict['all'][model_name].append(idx)

    with (RESULTS_DIR / 'rankings.csv').open(mode='w') as out_file:
        for dataset in (DATASETS + ('all',)):
            out_file.writelines([f'{dataset}\n'])
            print(f'\nRanking for {dataset}:\n')

            dset_ranking_dict = {name: sum(scores) / len(scores) for name, scores in ranking_dict[dataset].items()}
            ranked = tuple((name, score) for name, score in sorted(dset_ranking_dict.items(), key=lambda item: item[1]))
            for name, score in ranked:
                out_file.writelines([f'{score:.3f},  {name}\n'])
                print(f'{score:.3f}: {name}')


def plot_fig_15(df):
    fig, axes = plt.subplots(nrows=len(DATASETS), ncols=len(DIS_METRICS), figsize=(30, 20))

    for row_idx, dataset in enumerate(DATASETS):
        dset_df = get_dataset_df(df, dataset)

        for col_idx, metric in enumerate(DIS_METRICS):
            ax = axes[row_idx, col_idx]
            metric_df, metric_col_name = get_metric_df(dset_df, metric)
            x_ranges = []

            for method in METHODS:
                reg_col_name = get_reg_col_name(method)
                method_df = get_method_df(metric_df, method)

                grouped_df = method_df.groupby(reg_col_name)[metric_col_name].mean()

                x_range = list(range(len(grouped_df)))
                x_ranges.append(len(x_range))
                sns.lineplot(
                    x=x_range,
                    y=grouped_df.values,
                    ax=ax,
                    label=method,
                    linewidth=4,
                )

            # plot baseline
            method_df = get_method_df(metric_df, 'beta_vae')
            sns.lineplot(
                x=list(range(max(x_ranges))),
                y=method_df[metric_col_name].mean(),
                ax=ax,
                label='beta_vae',
                linewidth=4,
            )

            ax.get_legend().remove()
            ax.grid()
            ax_kwargs = {}
            if row_idx == 0:
                # TODO metric names
                ax_kwargs['title'] = metric.replace('evaluation_results.', '')

                if col_idx == len(DIS_METRICS) // 2:
                    # display the legend (just once)
                    # ax.legend(
                    #     loc='upper center',
                    #     bbox_to_anchor=(0, 1.5),
                    #     ncol=len(METHODS) + 1,
                    # )
                    ax.legend()
                    # handles, labels = ax.get_legend_handles_labels()

            if col_idx == 0:
                ax_kwargs['ylabel'] = 'Value'
            if row_idx == len(DATASETS) - 1:
                ax_kwargs['xlabel'] = 'Regularization strength'
            if col_idx == len(DIS_METRICS) - 1:
                ax.yaxis.set_label_position('right')
                ax_kwargs['ylabel'] = dataset

            ax.set(**ax_kwargs)

    # fig.subplots_adjust(top=0.9, left=0.1, right=0.9,
    #                     bottom=0.52)  # create some space below the plots by increasing the bottom-value
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.52), ncol=len(METHODS) + 1)

    # fig.subplots_adjust(top=2)
    # fig.legend(handles, labels,
    #               loc='upper center',
    #               bbox_to_anchor=(0, 1.5),
    #               # ncol=5,
    #               )
    plt.savefig(PLOT_DIR / 'fig_15.png')
    plt.show()
    plt.close(fig)


def plot_rank_corr_matrices(df):
    fig, axes = plt.subplots(nrows=len(METHODS), ncols=len(DIS_METRICS), figsize=(30, 20))

    for row_idx, method in enumerate(METHODS):
        method_df = get_method_df(df, method)
        reg_col_name = get_reg_col_name(method)

        for col_idx, metric in enumerate(DIS_METRICS):
            ax = axes[row_idx, col_idx]
            metric_df, metric_col_name = get_metric_df(method_df, metric)

            grouped_df = metric_df.groupby((DATASET_COL_STR, reg_col_name))[
                metric_col_name].mean().reset_index().sort_values(reg_col_name)
            grouped_df = pd.DataFrame({dataset: get_dataset_df(grouped_df, dataset)[metric_col_name].values for dataset in DATASETS}, dtype=np.float)
            print(grouped_df)
            corr = grouped_df.corr(method='spearman').fillna(0)

            sns.heatmap(
                data=corr,
                annot=True,
                cmap=sns.color_palette("coolwarm", 7),
                cbar=False,
                ax=ax,
            )

            ax_kwargs = {}
            if row_idx == 0:
                # TODO metric names
                ax_kwargs['title'] = metric.replace('evaluation_results.', '')
            if col_idx == len(DIS_METRICS) - 1:
                ax.yaxis.set_label_position('right')
                ax_kwargs['ylabel'] = method

            ax.set(**ax_kwargs)

    plt.savefig(PLOT_DIR / 'fig_17.png')
    plt.show()
    plt.close(fig)


def main():
    sweep = h.product((
        h.sweep('dataset', DATASETS),
        h.sweep('method', METHODS),
        h.sweep('beta', (
            # 1,
            # 4,
            # 8,
            16,
            # 32,
        )),
        h.sweep('all_layers', (
            True,
            # False,
        )),
        h.sweep('scale', (
            True,
            # False,
        )),
    ))

    df_list = []
    for setting in sweep:
        dataset, method, beta, all_layers, scale = setting['dataset'], setting['method'], setting['beta'], setting[
            'all_layers'], setting['scale']

        if 'masked' in method:
            scale = False

        out_dir = PLOT_DIR / (method + ('_all' if all_layers else '') + ('_scale' if scale else '')) / f'beta_{beta}'

        df = load_results()

        if 'dim_wise_mask_l1' in method:
            dim_wise_df = df.loc[df[MODEL_COL_STR].str.contains('dim_wise_mask_l1')]
            dim_wise_df[MODEL_COL_STR] = dim_wise_df.apply(
                lambda row: row[MODEL_COL_STR].replace('l1', 'l1_' + row[
                    'train_config.dim_wise_l1_vae.dim'].replace("'", '')),
                axis=1,
            )
            df.loc[df[MODEL_COL_STR].str.contains('dim_wise_mask_l1')] = dim_wise_df

        df = df.loc[df[MODEL_COL_STR].str.contains(method)]

        if 'dim_wise_mask_l1' in method:
            if not ('row' in method or 'col' in method):
                # non dim wise l1
                df = df.loc[df['train_config.dim_wise_l1_vae.dim'] == 'None']
        idxs_all_layers = (df['train_config.dim_wise_l1_vae.all_layers'] == 'True') | (
                df['train_config.conv_encoder.all_layers'] == 'True')
        df = df.loc[idxs_all_layers] if all_layers else df.loc[~idxs_all_layers]
        idxs_scale = (df['train_config.dim_wise_l1_vae.scale_per_layer'] == 'True')
        df = df.loc[idxs_scale] if scale else df.loc[~idxs_scale]

        print(dataset, out_dir.parts[1:])
        print(len(df))
        if len(df) < 1:
            continue

        if 'dim_wise' in method:
            # TODO nicer way of mapping reg. strengths
            df[MODEL_COL_STR] = df[MODEL_COL_STR] + '_' + df['train_config.dim_wise_l1_vae.lmbd_l1'].map(
                "{:.2e}".format)
            df[MODEL_COL_STR] = df[MODEL_COL_STR].str.replace('dim_wise_mask_l1_', '')
            df[MODEL_COL_STR] = df[MODEL_COL_STR].str.replace('_vae', '')
            # df = df.loc[df['train_config.dim_wise_l1_vae.lmbd_l1'].between(1e-10, 1e0)]
            df = df.loc[df['train_config.dim_wise_l1_vae.lmbd_l1'].isin((
                # 1e-10,
                # 3.1622776601683795e-10,
                # 1e-09,
                # 3.1622776601683795e-09,
                # 1e-08,
                # 3.162277660168379e-08,
                # 1e-07,
                # 4.641588833612782e-07,
                1e-06,
                # 3.162277660168379e-06,
                1e-05,
                # 3.727593720314938e-05,
                0.0001,
                # 0.00031622776601683794,
                0.001,
                # 0.0031622776601683794,
                0.01,
                # 0.03162277660168379,
                # TODO
                # 0.1,
                # 0.31622776601683794,
                # 1.0,
            ))]
            reg_weight_col = 'train_config.dim_wise_l1_vae.lmbd_l1'
        elif method == 'masked':
            df[MODEL_COL_STR] = df[MODEL_COL_STR] + '_' + df['train_config.conv_encoder.perc_sparse'].map(
                lambda x: round(float(x), 2)
            ).map(str)
            reg_weight_col = 'train_config.conv_encoder.perc_sparse'
            df[reg_weight_col] = pd.to_numeric(df[reg_weight_col])
        elif method == 'weight_decay':
            reg_weight_col = 'train_config.dim_wise_l1_vae.lmbd_l2'

        if dataset == 'shapes3d':
            dlib_df = load_results()
            dlib_df = dlib_df.loc[dlib_df[MODEL_COL_STR] == "'beta_vae'"]
        else:
            dlib_df = load_dlib_df()

        dlib_df[reg_weight_col] = 0
        df = pd.concat((df, dlib_df), sort=True)
        df = df.loc[
            (df['train_config.vae.beta'] == beta)
            & (df[DATASET_COL_STR] == f"'{dataset}'")
            ]
        if len(df[MODEL_COL_STR].unique()) < 2:
            continue
        df[MODEL_COL_STR] = df[MODEL_COL_STR].str.replace("'", '')

        # out_dir.mkdir(exist_ok=True, parents=True)
        # plot_results(
        #     df=df,
        #     dataset=dataset,
        #     out_dir=out_dir,
        #     reg_weight_col=reg_weight_col,
        # )

        df[MODEL_COL_STR] = out_dir.parent.name + '_' + df[MODEL_COL_STR]

        df = df.loc[df[reg_weight_col] != 0]
        df_list.append(df)

    df = pd.concat(df_list)
    shapes_baseline = load_results()
    shapes_baseline = shapes_baseline.loc[
        (shapes_baseline[DATASET_COL_STR].str.contains('shapes3d'))
        & (shapes_baseline[MODEL_COL_STR].str.contains('beta_vae'))
        ]
    df = pd.concat((df, load_dlib_df(), shapes_baseline))
    df = df.loc[df['train_config.vae.beta'] == 16]

    plot_rank_corr_matrices(df)
    # plot_fig_15(df)

    # print_rankings(df)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('result_files', type=str, nargs='+', )
    args = parser.parse_args()

    main()
