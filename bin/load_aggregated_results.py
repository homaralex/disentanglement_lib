import argparse

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

MODEL_COL_STR = 'train_config.model.name'

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# TODO all metrics in one plot
def plot_results(results_file):
    df = pd.read_json(results_file)

    for metric in (
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
            'evaluation_results.gaussian_wasserstein_correlation',
            'evaluation_results.gaussian_wasserstein_correlation_norm',
            'evaluation_results.mutual_info_score',
    ):
        print()
        if metric in ('beta_vae_sklearn', 'factor_vae_metric'):
            metric_df = df.loc[df['evaluation_config.evaluation.name'].str.contains(metric)]
            # print(metric_df[['train_config.model.name', 'evaluation_results.eval_accuracy']])
            metric = 'evaluation_results.eval_accuracy'
        else:
            metric_df = df.loc[df[metric].notna()]

        print(metric_df.groupby(MODEL_COL_STR)[metric].mean().reset_index().sort_values(metric, ascending=False))

        sns.violinplot(
            x=MODEL_COL_STR,
            y=metric,
            data=metric_df,
            cut=0,
        )
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('results_file', type=str)
    args = parser.parse_args()

    results_df = plot_results(args.results_file)
