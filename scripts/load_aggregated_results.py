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
    # 'beta_vae_sklearn',
    'factor_vae_metric',
    'evaluation_results.discrete_mig',
    'evaluation_results.disentanglement',
    'evaluation_results.modularity_score',
    'evaluation_results.SAP_score',

    # 'train_results.loss',
    # 'train_results.regularizer',

    'evaluation_results.completeness',
    'evaluation_results.informativeness_test',
    'evaluation_results.explicitness_score_test',

    # 'train_results.reconstruction_loss',

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
    'color_dsprites',
    'noisy_dsprites',
    'scream_dsprites',
    'shapes3d',
    'cars3d',
    'smallnorb',
)
METHODS = (
    # 'masked',
    # 'dim_wise_mask_l1_col',
    # 'dim_wise_mask_l1',
    'vd_vae',
    # 'softmax_vae',
    # 'wae',
    # 'proximal_vae',

    # 'small_vae',
    # 'weight_decay',
    # 'dim_wise_mask_l1_row',
)

HUMAN_READABLE_NAMES = {

    # methods
    'masked': 'RPM',
    'dim_wise_mask_l1_col': 'Dim-Wise LL1M',
    'dim_wise_mask_l1': 'LL1M',
    'vd_vae': 'VD',
    'proximal_vae': 'Proximal',
    'small_vae': 'Small-VAE',
    'weight_decay': 'Weight Decay',
    'beta_vae': 'BetaVAE',

    # datasets
    'dsprites_full': 'dSprites',
    'color_dsprites': 'Color-dSprites',
    'noisy_dsprites': 'Noisy-dSprites',
    'scream_dsprites': 'Scream-dSprites',
    'shapes3d': 'Shapes3D',
    'cars3d': 'Cars3D',
    'smallnorb': 'Norb',

    # unsup metrics
    'train_results.reconstruction_loss': 'Reconstruction',
    'train_results.kl_loss': 'KL',
    'train_results.elbo': 'ELBO',

    # dis metrics
    'beta_vae_sklearn': 'BetaVAE Score',
    'factor_vae_metric': 'FactorVAE Score',
    'evaluation_results.discrete_mig': 'MIG',
    'evaluation_results.disentanglement': 'DCI Disentanglement',
    'evaluation_results.modularity_score': 'Modularity',
    'evaluation_results.SAP_score': 'SAP Score',

    # 'completeness' metrics
    'evaluation_results.completeness': 'completeness',
    'evaluation_results.informativeness_test': 'informativeness',
    'evaluation_results.explicitness_score_test': 'explicitness',
}

PLOT_DIR = Path('plots')
PLOT_DIR.mkdir(exist_ok=True)


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
        h.sweep('anneal', (
            True,
            False,
        )),
    ))

    df_list = []
    model_names = set()
    for setting in sweep:
        dataset, method, beta, all_layers, scale, anneal = setting['dataset'], setting['method'], setting['beta'], \
                                                           setting[
                                                               'all_layers'], setting['scale'], setting['anneal']

        if 'masked' in method or 'small' in method or 'weight_decay' in method or 'proximal' in method or 'wae' in method:
            scale = False

        out_dir = PLOT_DIR / (method + ('_anneal' if anneal else '') + ('_all' if all_layers else '') + (
            '_scale' if scale else '')) / f'beta_{beta}'

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
        idxs_scale = (df[get_scale_col_name(method)] == 'True')
        df = df.loc[idxs_scale] if scale else df.loc[~idxs_scale]
        idxs_anneal = (~df['train_config.vd_vae.anneal_kld_for'].isin(('None', None)))
        df = df.loc[idxs_anneal] if anneal else df.loc[~idxs_anneal]

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
                0.1,
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
        # TODO use get_reg_col_name() fn here
        elif 'vd' in method:
            reg_weight_col = 'train_config.vd_vae.lmbd_kld_vd'
        elif 'proximal' in method:
            reg_weight_col = 'train_config.proximal_vae.lmbd_prox'
        elif 'softmax' in method:
            reg_weight_col = 'train_config.conv_encoder.softmax_temperature'
        elif 'wae' in method:
            reg_weight_col = 'train_config.vae.beta'
            new_reg_weight_col = reg_weight_col.replace('vae', 'wae')
            df[new_reg_weight_col] = df[reg_weight_col]
            df[reg_weight_col] = beta
            reg_weight_col = new_reg_weight_col
        # ablation test methods
        elif 'weight_decay' in method:
            reg_weight_col = 'train_config.dim_wise_l1_vae.lmbd_l2'
        elif 'small_vae' in method:
            reg_weight_col = 'train_config.conv_encoder.perc_units'

        if dataset == 'shapes3d':
            dlib_df = load_results()
            dlib_df = dlib_df.loc[dlib_df[MODEL_COL_STR] == "'beta_vae'"]
        else:
            dlib_df = load_dlib_df()

        model_name = out_dir.parent.name
        df[MODEL_COL_STR] = model_name
        model_names.add(model_name)

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

    top_k_groups = None
    # plot_fig_15(df, methods=model_names, top_k_groups=top_k_groups)
    # plot_fig_16(df, methods=METHODS[:3] + ('beta_vae',))
    # plot_fig_16(df, methods=[*model_names, 'beta_vae'], top_k_groups=top_k_groups)
    # plot_fig_16(df, methods=model_names, top_k_groups=top_k_groups, diff_over_baseline=True)
    # plot_fig_17(df, methods=METHODS[:3] + ('beta_vae',))
    # plot_fig_17(df, methods=[*model_names, 'beta_vae'], top_k_groups=top_k_groups)
    plot_fig_17(df, methods=model_names, top_k_groups=top_k_groups, diff_over_baseline=True)
    # plot_fig_18(df, methods=METHODS[:3])
    # plot_fig_18(df, methods=model_names, top_k_groups=top_k_groups)

    # print_rankings(df)


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
    elif 'beta_vae' in method or 'wae' in method:
        return 'train_config.vae.beta'
    elif 'weight_decay' in method:
        return 'train_config.dim_wise_l1_vae.lmbd_l2'
    elif 'small_vae' in method:
        return 'train_config.conv_encoder.perc_units'
    elif 'vd' in method:
        return 'train_config.vd_vae.lmbd_kld_vd'
    elif 'proximal' in method:
        return 'train_config.proximal_vae.lmbd_prox'
    elif 'softmax' in method:
        return 'train_config.conv_encoder.softmax_temperature'
    raise ValueError(method)


def get_scale_col_name(method):
    if 'softmax' in method:
        return 'train_config.conv_encoder.scale_temperature'
    if 'vd' in method:
        return 'train_config.vd_vae.scale_per_layer'
    else:
        return 'train_config.dim_wise_l1_vae.scale_per_layer'


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
            x=reg_weight_col,
            y=metric,
            data=metric_df,
            cut=0,
            ax=ax_violin,
        )
        for tick in ax_violin.get_xticklabels():
            tick.set_rotation(45)

        sns.boxplot(
            x=reg_weight_col,
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


def plot_fig_15(df, methods=METHODS, top_k_groups=None):
    fig, axes = plt.subplots(nrows=len(DATASETS), ncols=len(DIS_METRICS), figsize=(30, 20))

    for row_idx, dataset in enumerate(DATASETS):
        dset_df = get_dataset_df(df, dataset)

        for col_idx, metric in enumerate(DIS_METRICS):
            ax = axes[row_idx, col_idx]
            metric_df, metric_col_name = get_metric_df(dset_df, metric)
            x_ranges = []

            for method in methods:
                if 'weight_decay' in method or 'small' in method:
                    continue

                reg_col_name = get_reg_col_name(method)
                method_df = get_method_df(metric_df, method)

                grouped_df = method_df.groupby(reg_col_name)[metric_col_name].mean()
                if top_k_groups is not None:
                    grouped_df = grouped_df.iloc[:top_k_groups]

                x_range = list(range(len(grouped_df)))
                x_ranges.append(len(x_range))
                sns.lineplot(
                    # TODO remove ifs since we have multiple runs for small_vae now
                    x=(0, 5) if 'small' in method else x_range,
                    y=(grouped_df.iloc[::-1] if 'small' in method else grouped_df).values,
                    ax=ax,
                    # TODO change to readable
                    label=method,
                    # label=HUMAN_READABLE_NAMES[method],
                    linewidth=4,
                )

            # plot baselines
            method_df = get_method_df(metric_df, 'beta_vae')
            sns.lineplot(
                x=list(range(max(x_ranges))),
                y=method_df[metric_col_name].mean(),
                ax=ax,
                label=HUMAN_READABLE_NAMES['beta_vae'],
                zorder=10,
                color='black',
            )
            ax.lines[-1].set_linestyle((0, (5, 5)))
            method_df = get_method_df(metric_df, 'weight_decay')
            sns.lineplot(
                x=list(range(max(x_ranges))),
                y=method_df[metric_col_name].mean(),
                ax=ax,
                label=HUMAN_READABLE_NAMES['weight_decay'],
                linewidth=3,
                color='goldenrod',
                zorder=0,
            )
            for perc_units, color in ((.5, 'crimson'), (.75, 'violet')):
                method_df = get_method_df(metric_df, 'small_vae')
                method_df = method_df.loc[method_df[get_reg_col_name('small_vae')] == perc_units]
                sns.lineplot(
                    x=list(range(max(x_ranges))),
                    y=method_df[metric_col_name].mean(),
                    ax=ax,
                    label=f'{HUMAN_READABLE_NAMES["small_vae"]} {perc_units}',
                    linewidth=3,
                    color=color,
                    zorder=0,
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

    plt.savefig(PLOT_DIR / 'fig_15.png')
    plt.show()
    plt.close(fig)


def plot_fig_16(df, methods=METHODS, top_k_groups=None, diff_over_baseline=False):
    fig, axes = plt.subplots(nrows=1, ncols=len(methods), figsize=(30, 12))

    def get_corr_matrix(method):
        method_df = get_method_df(df, method)

        if top_k_groups is not None:
            reg_col_name = get_reg_col_name(method)
            top_k_reg_vals = sorted(method_df[reg_col_name].unique())[:top_k_groups]
            method_df = method_df.loc[method_df[reg_col_name].isin(top_k_reg_vals)]

        corr_matrix = np.empty(shape=(len(DIS_METRICS), len(UNSUP_METRICS)), dtype=np.float)
        for corr_row_idx, metric in enumerate(DIS_METRICS):
            metric_df, metric_col_name = get_metric_df(method_df, metric)

            for corr_col_idx, unsup_metric in enumerate(UNSUP_METRICS):
                metric_pair_df = method_df[[unsup_metric, metric_col_name]]

                corr = metric_pair_df.corr(method='pearson')
                corr_matrix[corr_row_idx, corr_col_idx] = corr.iat[0, 1]

        return corr_matrix

    baseline_corr_matrix = get_corr_matrix('beta_vae')

    for col_idx, method in enumerate(methods):
        ax = axes[col_idx]
        corr_matrix = get_corr_matrix(method)
        if diff_over_baseline:
            corr_matrix = np.abs(corr_matrix) - np.abs(baseline_corr_matrix)

        h = sns.heatmap(
            data=corr_matrix,
            annot=True,
            vmin=-1,
            vmax=1,
            cmap=sns.color_palette("coolwarm", 7),
            cbar=False,
            square=True,
            yticklabels=col_idx == 0,
            ax=ax,
        )
        h.set_xticklabels(
            tuple(HUMAN_READABLE_NAMES[metric_name] for metric_name in UNSUP_METRICS),
            rotation=90,
        )
        if col_idx == 0:
            h.set_yticklabels(
                tuple(HUMAN_READABLE_NAMES[metric_name] for metric_name in DIS_METRICS),
                rotation=0,
            )

        # ax.set_title(HUMAN_READABLE_NAMES[method])
        ax.set_title(method)

    plt.savefig(PLOT_DIR / 'fig_16.png')
    plt.show()
    plt.close(fig)


def plot_fig_17(df, methods=METHODS, top_k_groups=None, diff_over_baseline=False):
    # methods = METHODS + ('beta_vae',)
    fig, axes = plt.subplots(nrows=len(methods), ncols=len(DIS_METRICS), figsize=(30, 20))

    def get_corr_matrix(method):
        method_df = get_method_df(df, method)
        reg_col_name = get_reg_col_name(method)
        metric_df, metric_col_name = get_metric_df(method_df, metric)

        if 'beta_vae' in method:
            shapes3d_df = pd.concat([get_dataset_df(metric_df, 'shapes3d')]).head(50)
            metric_df = pd.concat([metric_df.loc[metric_df[DATASET_COL_STR] != 'shapes3d'], shapes3d_df])
            grouped_df = metric_df
            grouped_df = pd.DataFrame(
                {dataset: get_dataset_df(grouped_df, dataset)[metric_col_name].head(7).values for dataset in
                 DATASETS},
                dtype=np.float)
        else:
            grouped_df = metric_df.groupby((DATASET_COL_STR, reg_col_name))[metric_col_name].mean().reset_index()

            grouped_df = pd.DataFrame(
                {dataset: get_dataset_df(grouped_df, dataset).sort_values(reg_col_name)[metric_col_name].values
                 for
                 dataset in DATASETS
                 },
                dtype=np.float,
            )

            # grouped_df = pd.concat(tuple(
            #     metric_df.loc[metric_df[reg_col_name] == reg_val].sort_values(DATASET_COL_STR).head(12) for reg_val
            #     in metric_df[reg_col_name].unique()))

            # slices = []
            # for dset in DATASETS:
            #     for reg_val in metric_df[reg_col_name].unique():
            #         slice_df = get_dataset_df(metric_df, dset)
            #         slice_df = slice_df.loc[slice_df[reg_col_name] == reg_val]
            #         slices.append(slice_df)
            # min_len = min(map(len, slices))
            # grouped_df = pd.concat(tuple(slice.head(min_len) for slice in slices))
            #
            # grouped_df = pd.DataFrame(
            #     {dataset: get_dataset_df(grouped_df, dataset).sort_values(reg_col_name)[metric_col_name].values for
            #      dataset in DATASETS},
            #     dtype=np.float)

        for dataset in DATASETS:
            if top_k_groups is not None and 'beta_vae' not in method:
                grouped_df[dataset] = grouped_df[dataset].iloc[:top_k_groups]

            # prevent zero-division error in case of no variance by artificially changing one entry slightly
            if grouped_df[dataset].std() == 0:
                grouped_df[dataset][0] = grouped_df[dataset][0] * .999999

        corr = grouped_df.corr(method='pearson')

        corr.rename(
            columns=lambda c: HUMAN_READABLE_NAMES[c],
            index=lambda i: HUMAN_READABLE_NAMES[i],
            inplace=True,
        )

        return corr

    # mean_corrs = np.zeros((len(methods), len(DIS_METRICS)))
    mean_corrs = defaultdict(dict)
    for row_idx, method in enumerate(methods):
        for col_idx, metric in enumerate(DIS_METRICS):
            ax = axes[row_idx, col_idx]

            corr = get_corr_matrix(method=method)
            if diff_over_baseline:
                baseline_corr = get_corr_matrix('beta_vae')
                corr = (corr - baseline_corr) / 2

            # mean_corrs[row_idx, col_idx] = np.array(corr).mean()
            mean_corrs[method][metric] = (
                        np.tril(np.array(corr)).sum() / (len(methods) - 1) / (len(DIS_METRICS) - 1)).round(2)

            sns.heatmap(
                data=corr,
                annot=True,
                vmin=-1,
                vmax=1,
                cmap=sns.color_palette("coolwarm", 7),
                cbar=False,
                xticklabels=(row_idx == len(methods) - 1),
                yticklabels=(col_idx == 0),
                ax=ax,
            )

            ax_kwargs = {}
            if row_idx == 0:
                ax_kwargs['title'] = HUMAN_READABLE_NAMES[metric]
            if col_idx == len(DIS_METRICS) - 1:
                ax.yaxis.set_label_position('right')
                # TODO change to human readable
                ax_kwargs['ylabel'] = method  # HUMAN_READABLE_NAMES[method]

            ax.set(**ax_kwargs)

    # print(mean_corrs.T)
    print(pd.DataFrame(mean_corrs))

    plt.savefig(PLOT_DIR / 'fig_17.png')
    plt.show()
    plt.close(fig)


def plot_fig_18(df, methods=METHODS, top_k_groups=None):
    datasets = DATASETS[1:]
    # datasets = (DATASETS[0], ) + DATASETS[2:]
    fig, axes = plt.subplots(nrows=len(datasets), ncols=len(DIS_METRICS), figsize=(30, 20))

    for col_idx, metric in enumerate(DIS_METRICS):
        metric_df, metric_col_name = get_metric_df(df, metric)
        dsprites_df = get_dataset_df(metric_df, 'dsprites_full')
        # dsprites_df = get_dataset_df(metric_df, 'scream_dsprites')

        for row_idx, dataset in enumerate(datasets):
            ax = axes[row_idx, col_idx]
            dataset_df = get_dataset_df(metric_df, dataset)

            for method in methods:
                reg_col_name = get_reg_col_name(method)
                method_df = get_method_df(dataset_df, method).groupby(reg_col_name)[metric_col_name].mean()
                dsprites_method_df = get_method_df(dsprites_df, method).groupby(reg_col_name)[metric_col_name].mean()
                if top_k_groups is not None:
                    method_df, dsprites_method_df = method_df.iloc[:top_k_groups], dsprites_method_df.iloc[
                                                                                   :top_k_groups]

                sns.scatterplot(
                    x=dsprites_method_df.values,
                    y=method_df.values,
                    # label=HUMAN_READABLE_NAMES[method],
                    # TODO change back to human readable
                    label=method,
                    s=128,
                    # marker=('o' if 'col' in method else ('^' if 'l1' in method else 's')),
                    ax=ax,
                )

            ax.get_legend().remove()
            ax.grid()
            ax_kwargs = {}
            if row_idx == 0:
                ax_kwargs['title'] = HUMAN_READABLE_NAMES[metric]
            if col_idx == 0:
                ax_kwargs['ylabel'] = 'Value'
            if row_idx == len(datasets) - 1:
                ax_kwargs['xlabel'] = 'dSprites'
            if col_idx == len(DIS_METRICS) - 1:
                ax.yaxis.set_label_position('right')
                ax_kwargs['ylabel'] = HUMAN_READABLE_NAMES[dataset]

            ax.set(**ax_kwargs)

    axes[0, 2].legend(loc='upper center', bbox_to_anchor=(1., 1.6), ncol=len(methods) + 1)

    plt.savefig(PLOT_DIR / 'fig_18.png')
    plt.show()
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('result_files', type=str, nargs='+', )
    args = parser.parse_args()

    main()
