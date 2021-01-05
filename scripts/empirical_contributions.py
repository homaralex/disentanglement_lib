import numpy as np

from disentanglement_lib.data.ground_truth import named_data
from disentanglement_lib.data.ground_truth import util

num_sampled_factors = 200
random_state = np.random.RandomState(1)


def estimate_gradient(dataset):
    factors = dataset.sample_factors(num=num_sampled_factors, random_state=random_state)
    batch = dataset.sample_observations_from_factors(factors=factors, random_state=random_state)

    batch_squeezed = np.reshape(batch, (batch.shape[0], -1))

    for dim_idx in range(batch_squeezed.shape[1]):
        f = batch_squeezed[:, dim_idx]
        f = np.expand_dims(f, 1)
        grad = np.gradient(
            f,
            factors,
        )
        print(grad)


def estimate_contributions(dataset):
    sampled_factors = dataset.state_space.sample_latent_factors(num=num_sampled_factors, random_state=random_state)
    factor_contribs = np.zeros((len(dataset.latent_factor_indices),))

    for factor_idx, factor_size in enumerate(np.array(dataset.factor_sizes)[dataset.latent_factor_indices]):
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

    return factor_contribs


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

    estimated_contribs = estimate_contributions(dataset)
    # print(estimated_contribs.round())
    normalized_contribs = estimated_contribs / np.linalg.norm(estimated_contribs, 2)
    print(normalized_contribs.round(2))
    print(normalized_contribs.std().round(3))
