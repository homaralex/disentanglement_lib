import argparse
import os
from collections import defaultdict

import numpy as np
import gin.tf
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import gfile

from disentanglement_lib.data.ground_truth import named_data
from disentanglement_lib.utils import results


def main(
        model_dir,
        # output_dir,
        num_points,
        overwrite=True,
):
    # Fix the random seed for reproducibility.
    random_state = np.random.RandomState(0)

    # # Create the output directory if necessary.
    # if tf.gfile.IsDirectory(output_dir):
    #     if overwrite:
    #         tf.gfile.DeleteRecursively(output_dir)
    #     else:
    #         raise ValueError("Directory already exists and overwrite is False.")

    # Automatically set the proper data set if necessary. We replace the active
    # gin config as this will lead to a valid gin config file where the data set
    # is present.
    # Obtain the dataset name from the gin config of the previous step.
    gin_config_file = os.path.join(model_dir, 'results', 'gin', 'train.gin')
    gin_dict = results.gin_dict(gin_config_file)
    gin.bind_parameter('dataset.name', gin_dict['dataset.name'].replace("'", ""))

    dataset = named_data.get_named_ground_truth_data()
    module_path = os.path.join(model_dir, 'tfhub')

    with hub.eval_function_for_module(module_path) as f:
        sampled_latents = dataset.state_space.sample_latent_factors(
            num=num_points,
            random_state=random_state,
        )
        sampled_points = dataset.sample_all_factors(
            latent_factors=sampled_latents,
            random_state=random_state,
        )

        # encode 'real' images
        real_pics = dataset.sample_observations_from_all_factors(factors=sampled_points, random_state=random_state)

        encodings = batched_encoder(f, images=real_pics)

        means = encodings['mean']
        logvars = encodings['logvar']

        print(means.std(axis=0).round(3))
        print(np.exp(logvars).mean(axis=0).round(3))

        # investigate factor-encoding relations
        per_factor_encodings = defaultdict(list)
        for factor_idx, factor_size in enumerate(dataset.all_factor_sizes):
            xs = np.arange(0, factor_size, 1)
            if factor_size < 2:
                continue

            for curr_point in sampled_points:
                curr_point = np.expand_dims(curr_point.copy(), axis=0)
                points = np.repeat(curr_point, factor_size, axis=0)
                points[:, factor_idx] = xs

                ys = dataset.sample_observations_from_all_factors(factors=points, random_state=random_state)
                zs = batched_encoder(f, images=ys)['mean']

                per_factor_encodings[factor_idx].append(zs)

        per_factor_stds = np.zeros((len(dataset.all_factor_sizes), means.shape[1]))
        for factor_idx, factor_encodings in per_factor_encodings.items():
            per_factor_stds[factor_idx] = np.array(factor_encodings).std(axis=1).mean(axis=0)

        for factor_idx, factor_std in enumerate(per_factor_stds):
            print(factor_idx, factor_std.round(3))


def batched_encoder(
        model_fn,
        images,
        batch_size=64,
):
    # Push images through the TFHub module.
    num_outputs = 0
    output = []
    while num_outputs < images.shape[0]:
        inputs = images[num_outputs:min(num_outputs + batch_size, images.shape[0])]
        output_batch = model_fn(dict(images=inputs), signature='gaussian_encoder', as_dict=True)
        num_outputs += batch_size
        output.append(output_batch)

    output = {key: np.concatenate([batch[key] for batch in output]) for key in output[0]}

    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--num_points', type=int, default=64)
    args = parser.parse_args()

    main(
        model_dir=args.model_dir,
        # output_dir=args.output_dir,
        num_points=args.num_points,
    )
