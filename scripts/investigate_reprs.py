import json
import argparse
from collections import defaultdict
from pathlib import Path

import gin.tf
import numpy as np
import pandas as pd
import tensorflow_hub as hub

from disentanglement_lib.data.ground_truth import named_data
from disentanglement_lib.utils import results


def main(
        model_dir,
        output_file,
        num_points,
):
    model_dir = Path(model_dir)
    netstore_prefix = Path('../netstore') / 'sparse_dlib' / 'sparse_results'
    # Fix the random seed for reproducibility.
    random_state = np.random.RandomState(0)

    # Automatically set the proper data set if necessary. We replace the active
    # gin config as this will lead to a valid gin config file where the data set
    # is present.
    # Obtain the dataset name from the gin config of the previous step.
    gin_config_file = model_dir / 'model' / 'results' / 'gin' / 'train.gin'
    module_path = model_dir / 'model' / 'tfhub'

    if not gin_config_file.exists():
        gin_config_file = netstore_prefix / gin_config_file
        module_path = netstore_prefix / module_path

    gin_dict = results.gin_dict(str(gin_config_file))
    gin.bind_parameter('dataset.name', gin_dict['dataset.name'].replace("'", ""))

    dataset = named_data.get_named_ground_truth_data()

    with hub.eval_function_for_module(str(module_path)) as f:
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

        means_std = means.std(axis=0)
        avg_variances = np.exp(logvars).mean(axis=0)

        print(means_std.round(3))
        print(avg_variances.round(3))

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

    results_path = model_dir / 'postprocessed' / 'mean' / 'results' / 'aggregate' / 'train.json'
    json_results = json.load(results_path.open())
    uuid = json_results['train_results.uuid']

    results_row = {
        'uuid': uuid,
        'per_factor_stds': per_factor_stds,
        'avg_means': means_std,
        'variances': avg_variances,
    }
    print(results_row)

    output_file = Path(output_file)
    curr_row = pd.DataFrame([results_row])
    if output_file.exists():
        df = pd.read_pickle(output_file)
        df = pd.concat((df, curr_row))
    else:
        df = curr_row
    df.to_pickle(output_file)

    print(df.head())


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
    parser.add_argument('--output_file', type=str, default='investigate_reprs.pkl')
    parser.add_argument('--num_points', type=int, default=64)
    args = parser.parse_args()

    main(
        model_dir=args.model_dir,
        output_file=args.output_file,
        num_points=args.num_points,
    )
