import argparse

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

MODEL_COL_STR = 'train_config.model.name'

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

METRICS = (
    'evaluation_results.informativeness_test',
    'evaluation_results.disentanglement',
    'evaluation_results.completeness',
    'evaluation_results.discrete_mig',
    # 'evaluation_results.eval_accuracy',
    'beta_vae_sklearn',
    'factor_vae_metric',
    'evaluation_results.modularity_score',
    'evaluation_results.explicitness_score_test',
    # 'evaluation_results.num_active_dims',
    'evaluation_results.SAP_score',
    'evaluation_results.gaussian_total_correlation',
    # 'evaluation_results.gaussian_wasserstein_correlation',
    'evaluation_results.gaussian_wasserstein_correlation_norm',
    'evaluation_results.mutual_info_score',
)


def plot_results(results_file):
    df = pd.read_json(results_file)

    # TODO nicer way of mapping reg. strengths
    df[MODEL_COL_STR] = df[MODEL_COL_STR] + '_' + df['train_config.dim_wise_l1_vae.lmbd_l1'].map("{:.2e}".format)
    df[MODEL_COL_STR] = df[MODEL_COL_STR].str.replace('dim_wise_l1_', '')
    df[MODEL_COL_STR] = df[MODEL_COL_STR].str.replace('_nan', '')
    df[MODEL_COL_STR] = df[MODEL_COL_STR].str.replace('_vae', '')

    fig_violin, axes_violin = plt.subplots(nrows=3, ncols=4, figsize=(30, 30))
    fig_box, axes_box = plt.subplots(nrows=3, ncols=4, figsize=(30, 30))
    fig_mean, axes_mean = plt.subplots(nrows=3, ncols=4, figsize=(30, 30))
    for metric, ax_violin, ax_box, ax_mean in zip(
            METRICS,
            axes_violin.flatten(),
            axes_box.flatten(),
            axes_mean.flatten(),
    ):
        print()
        if metric in ('beta_vae_sklearn', 'factor_vae_metric'):
            metric_df = df.loc[df['evaluation_config.evaluation.name'].str.contains(metric)]
            # print(metric_df[['train_config.model.name', 'evaluation_results.eval_accuracy']])
            metric = 'evaluation_results.eval_accuracy'
        else:
            metric_df = df.loc[df[metric].notna()]

        print(metric_df.groupby(MODEL_COL_STR)[metric].mean().reset_index().sort_values(metric, ascending=False))

        metric_df = metric_df.sort_values(MODEL_COL_STR)

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
        metric_df = metric_df.groupby(MODEL_COL_STR)[metric].mean()
        sns.stripplot(
            x=metric_df.index,
            y=metric_df,
            ax=ax_mean,
            size=25,
        )
        for tick in ax_mean.get_xticklabels():
            tick.set_rotation(45)

    fig_violin.savefig('plots/violin.png')
    fig_box.savefig('plots/box.png')
    fig_mean.savefig('plots/mean.png')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('results_file', type=str)
    args = parser.parse_args()

    results_df = plot_results(args.results_file)
