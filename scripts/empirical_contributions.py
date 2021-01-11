import argparse

import numpy as np

from disentanglement_lib.data.ground_truth import named_data
from disentanglement_lib.data.ground_truth import util


def estimate_contributions(
        dataset,
        num_sampled_factors,
        random_state,
        normalize=True,
):
    factor_contribs = np.zeros((len(dataset.factor_sizes),))
    sampled_latents = dataset.state_space.sample_latent_factors(
        num=num_sampled_factors,
        random_state=random_state,
    )
    sampled_factors = dataset.state_space.sample_all_factors(
        latent_factors=sampled_latents,
        random_state=random_state,
    )

    for factor_idx, factor_size in enumerate(dataset.factor_sizes):
        xs = np.arange(0, factor_size, 1)

        for curr_point in sampled_factors:
            curr_point = np.expand_dims(curr_point.copy(), axis=0)
            points = np.repeat(curr_point, factor_size, axis=0)
            points[:, factor_idx] = xs

            ys = dataset.sample_observations_from_factors(factors=points, random_state=random_state)
            diffs = np.diff(ys, axis=0)

            factor_contribs[factor_idx] += np.power(diffs, 2).sum()

        factor_contribs[factor_idx] /= sampled_factors.shape[0]
        factor_contribs[factor_idx] /= factor_size

    # filter out non-contributing dimensions
    factor_contribs = factor_contribs[factor_contribs != 0]

    if normalize:
        factor_contribs = factor_contribs / np.linalg.norm(factor_contribs, 2)

    return factor_contribs


def main(num_sampled_factors):
    random_state = np.random.RandomState(1)

    for dataset_name in (
            'dsprites_full',
            "color_dsprites",
            "noisy_dsprites",
            "scream_dsprites",
            "smallnorb",
            "cars3d",
            # TODO uncomment
            # "shapes3d",
    ):
        dataset = named_data.get_named_ground_truth_data(dataset_name)
        print(dataset.num_factors, dataset.intrinsic_num_factors, dataset_name)

        assert type(dataset.state_space) == util.SplitDiscreteStateSpace

        estimated_contribs = estimate_contributions(
            dataset=dataset,
            num_sampled_factors=num_sampled_factors,
            random_state=random_state,
        )

        # TODO save to file
        std_all = estimated_contribs.std()
        std = std_all
        if dataset.intrinsic_num_factors != dataset.num_factors:
            std = estimated_contribs[dataset.latent_factor_indices].std()

        print(estimated_contribs.round(3))
        print(std_all.round(3))
        print(std.round(3))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--num_sampled_factors', type=int, default=500)
    parser.add_argument('--intrinsic', action='store_true')
    args = parser.parse_args()

    main(num_sampled_factors=args.num_sampled_factors)
