import csv
import argparse
from pathlib import Path

import gin
import numpy as np

from disentanglement_lib.data.ground_truth import named_data
from disentanglement_lib.data.ground_truth import util

PLOT_DIR = Path('plots')
PLOT_DIR.mkdir(exist_ok=True)


def estimate_contributions(
        dataset,
        num_sampled_factors,
        random_state,
        normalize=True,
):
    factor_diffs = np.zeros((len(dataset.all_factor_sizes),))
    factor_variances = np.zeros((len(dataset.all_factor_sizes),))

    sampled_latents = dataset.state_space.sample_latent_factors(
        num=num_sampled_factors,
        random_state=random_state,
    )
    sampled_points = dataset.sample_all_factors(
        latent_factors=sampled_latents,
        random_state=random_state,
    )

    for factor_idx, factor_size in enumerate(dataset.all_factor_sizes):
        xs = np.arange(0, factor_size, 1)
        if factor_size < 2:
            continue

        for curr_point in sampled_points:
            curr_point = np.expand_dims(curr_point.copy(), axis=0)
            points = np.repeat(curr_point, factor_size, axis=0)
            points[:, factor_idx] = xs

            ys = dataset.sample_observations_from_all_factors(factors=points, random_state=random_state)
            diffs = np.diff(ys, axis=0)
            variances = ys.var(axis=0)

            factor_diffs[factor_idx] += np.power(diffs, 2).mean()
            factor_variances[factor_idx] += variances.mean()

        factor_diffs[factor_idx] /= sampled_points.shape[0]
        factor_variances[factor_idx] /= sampled_points.shape[0]

    # filter out non-contributing dimensions (i.e., hack for dsprites_full)
    factor_diffs = factor_diffs[factor_diffs != 0]
    factor_variances = factor_variances[factor_variances != 0]

    if normalize:
        factor_diffs = factor_diffs / np.linalg.norm(factor_diffs, 2)
        factor_variances = factor_variances / np.linalg.norm(factor_variances, 2)

    return factor_diffs, factor_variances


def main(
        out_file,
        num_sampled_factors,
        full_color_dims,
):
    random_state = np.random.RandomState(1)
    gin.bind_parameter('AbstractColorDSprites.color_as_single_dim', not full_color_dims)

    out_file = PLOT_DIR / out_file
    with out_file.open(mode='w') as csv_file:
        csv_writer = csv.writer(csv_file)
        # write header
        csv_writer.writerow([
            'dataset',
            'diffs',
            'diffs_all',
            'variances',
            'variances_all',
        ])

        for dataset_name in (
                'dsprites_full',
                'color_dsprites',
                'noisy_dsprites',
                'scream_dsprites',
                'smallnorb',
                'cars3d',
                'shapes3d',
        ):
            dataset = named_data.get_named_ground_truth_data(dataset_name)
            print(dataset.num_factors, dataset.intrinsic_num_factors, dataset_name)

            assert type(dataset.state_space) == util.SplitDiscreteStateSpace

            estimated_diffs, estimated_variances = estimate_contributions(
                dataset=dataset,
                num_sampled_factors=num_sampled_factors,
                random_state=random_state,
            )

            estimated_diffs_all, estimated_variances_all = estimated_diffs, estimated_variances
            if dataset.intrinsic_num_factors != dataset.num_factors:
                estimated_diffs_all = estimated_diffs[dataset.latent_factor_indices]
                estimated_variances_all = estimated_variances[dataset.latent_factor_indices]

            csv_writer.writerow([
                dataset_name,
                estimated_diffs,
                estimated_diffs_all,
                estimated_variances,
                estimated_variances_all,
            ])

            print(estimated_diffs.std().round(3), estimated_diffs.round(3))
            print(estimated_diffs_all.std().round(3), estimated_diffs_all.round(3))
            print(estimated_variances.std().round(3), estimated_variances.round(3))
            print(estimated_variances_all.std().round(3), estimated_variances_all.round(3))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--num_sampled_factors', type=int, default=100)
    parser.add_argument('--full_color_dims', action='store_true')
    parser.add_argument('--out_file', type=str, default='contribs.csv')
    args = parser.parse_args()

    main(
        out_file=args.out_file,
        num_sampled_factors=args.num_sampled_factors,
        full_color_dims=args.full_color_dims,
    )
