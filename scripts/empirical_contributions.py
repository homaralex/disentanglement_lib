import csv
import argparse
from pathlib import Path

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
    factor_contribs = np.zeros((len(dataset.all_factor_sizes),))
    sampled_latents = dataset.state_space.sample_latent_factors(
        num=num_sampled_factors,
        random_state=random_state,
    )
    sampled_factors = dataset.state_space.sample_all_factors(
        latent_factors=sampled_latents,
        random_state=random_state,
    )

    for factor_idx, factor_size in enumerate(dataset.all_factor_sizes):
        xs = np.arange(0, factor_size, 1)

        for curr_point in sampled_factors:
            curr_point = np.expand_dims(curr_point.copy(), axis=0)
            points = np.repeat(curr_point, factor_size, axis=0)
            points[:, factor_idx] = xs

            ys = dataset.sample_observations_from_all_factors(factors=points, random_state=random_state)
            diffs = np.diff(ys, axis=0)

            factor_contribs[factor_idx] += np.power(diffs, 2).sum()

        factor_contribs[factor_idx] /= sampled_factors.shape[0]
        factor_contribs[factor_idx] /= factor_size

    # filter out non-contributing dimensions (i.e., hack for dsprites_full)
    factor_contribs = factor_contribs[factor_contribs != 0]

    if normalize:
        factor_contribs = factor_contribs / np.linalg.norm(factor_contribs, 2)

    return factor_contribs


def main(out_file, num_sampled_factors):
    random_state = np.random.RandomState(1)

    out_file = PLOT_DIR / out_file
    with out_file.open(mode='w') as csv_file:
        csv_writer = csv.writer(csv_file)
        # write header
        csv_writer.writerow([
            'dataset',
            'all_std',
            'std',
            'all_contribs',
            'contribs',
        ])

        for dataset_name in (
                'dsprites_full',
                'color_dsprites',
                'noisy_dsprites',
                'scream_dsprites',
                'smallnorb',
                'cars3d',
                # TODO uncomment
                # 'shapes3d',
        ):
            dataset = named_data.get_named_ground_truth_data(dataset_name)
            print(dataset.num_factors, dataset.intrinsic_num_factors, dataset_name)

            assert type(dataset.state_space) == util.SplitDiscreteStateSpace

            estimated_contribs = estimate_contributions(
                dataset=dataset,
                num_sampled_factors=num_sampled_factors,
                random_state=random_state,
            )

            std_all = estimated_contribs.std()
            std = std_all
            if dataset.intrinsic_num_factors != dataset.num_factors:
                std = estimated_contribs[dataset.latent_factor_indices].std()

            csv_writer.writerow([
                dataset_name,
                std_all,
                std,
                estimated_contribs,
                estimated_contribs[dataset.latent_factor_indices],
            ])

            print(estimated_contribs.round(3))
            print(std_all.round(3))
            print(std.round(3))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    # TODO change default value
    parser.add_argument('--num_sampled_factors', type=int, default=20)
    parser.add_argument('--out_file', type=str, default='contribs.csv')
    args = parser.parse_args()

    main(
        out_file=args.out_file,
        num_sampled_factors=args.num_sampled_factors,
    )
