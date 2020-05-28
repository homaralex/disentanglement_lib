import argparse
import itertools
from collections import defaultdict
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import disentanglement_lib.utils.hyperparams as h

MODEL_COL_STR = 'train_config.model.name'
RESULTS_DIR = Path('aggregated_results')
DLIB_RESULTS_PATH = RESULTS_DIR / 'dlib_beta_vae_results.json'

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

METRICS = (
    'beta_vae_sklearn',
    'factor_vae_metric',
    'evaluation_results.discrete_mig',
    'evaluation_results.modularity_score',
    'evaluation_results.SAP_score',
    'evaluation_results.disentanglement',
    # 'evaluation_results.completeness',
    # 'evaluation_results.informativeness_test',
    # 'evaluation_results.explicitness_score_test',
    'evaluation_results.mutual_info_score',
    # 'evaluation_results.gaussian_total_correlation',
    # 'evaluation_results.gaussian_wasserstein_correlation_norm',
)
DATASETS = (
    'dsprites_full',
    'cars3d',
)
PLOT_DIR = Path('plots')
PLOT_DIR.mkdir(exist_ok=True)


def main(result_files):
    sweep = h.product((
        h.sweep('dataset', DATASETS),
        h.sweep('method', (
            'masked',
            'dim_wise_mask_l1_col',
            'dim_wise_mask_l1_row',
            'dim_wise_mask_l1',
            # 'weight_decay',
        )),
        h.sweep('beta', (
            # 1,
            # 4,
            # 8,
            16,
            # 32,
        )),
        h.sweep('all_layers', (True, False)),
        h.sweep('scale', (True, False)),
    ))
    partial_dfs = defaultdict(list)

    model_idx = 0
    for setting in sweep:
        dataset, method, beta, all_layers, scale = setting['dataset'], setting['method'], setting['beta'], setting[
            'all_layers'], setting['scale']

        out_dir = PLOT_DIR / (method + ('_all' if all_layers else '') + ('_scale' if scale else '')) / f'beta_{beta}'
        if scale and 'masked' in method:
            continue

        df = pd.concat((pd.read_json(RESULTS_DIR / results_file) for results_file in result_files), sort=True)

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
            df = df.loc[df['train_config.dim_wise_l1_vae.lmbd_l1'].between(1e-10, 1e0)]
            reg_weight_col = 'train_config.dim_wise_l1_vae.lmbd_l1'
        elif method == 'masked':
            df[MODEL_COL_STR] = df[MODEL_COL_STR] + '_' + df['train_config.conv_encoder.perc_sparse'].map(
                lambda x: round(float(x), 2)
            ).map(str)
            reg_weight_col = 'train_config.conv_encoder.perc_sparse'
            df[reg_weight_col] = pd.to_numeric(df[reg_weight_col])
        elif method == 'weight_decay':
            reg_weight_col = 'train_config.dim_wise_l1_vae.lmbd_l2'

        dlib_df = pd.read_json(DLIB_RESULTS_PATH)
        dlib_df[reg_weight_col] = 0
        df = pd.concat((df, dlib_df), sort=True)
        df = df.loc[
            (df['train_config.vae.beta'] == beta)
            & (df['train_config.dataset.name'] == f"'{dataset}'")
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

        # df[MODEL_COL_STR] = out_dir.parent.name + '_' + df[MODEL_COL_STR]
        df['model_idx'] = model_idx
        model_idx += 1
        partial_dfs[dataset].append(df.loc[df[reg_weight_col] != 0])

    for dataset in DATASETS:
        print(dataset)
        df = pd.concat(partial_dfs[dataset], sort=True)

        for metric in METRICS:
            metric_df, metric = get_metric_df(df, metric)
            print()
            print(metric_df.groupby('model_idx')[metric].mean().reset_index().sort_values(metric, ascending=False)[:5])

    df = pd.concat(itertools.chain(df for df in partial_dfs.values()))
    print(df.groupby('model_idx')[MODEL_COL_STR])


def get_metric_df(df, metric):
    if metric in ('beta_vae_sklearn', 'factor_vae_metric'):
        metric_df = df.loc[df['evaluation_config.evaluation.name'].str.contains(metric)]
        metric = 'evaluation_results.eval_accuracy'
    else:
        metric_df = df.loc[df[metric].notna()]

    return metric_df, metric


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
            METRICS,
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

        # group and aggreagate to obtain means per model
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('result_files', type=str, nargs='+', )
    args = parser.parse_args()

    main(args.result_files)
