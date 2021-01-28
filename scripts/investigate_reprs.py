import argparse
import os

import numpy as np
import gin.tf
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import gfile

from disentanglement_lib.visualize.visualize_model import sigmoid, tanh
from disentanglement_lib.data.ground_truth import named_data
from disentanglement_lib.utils import results


def main(
        model_dir,
        # output_dir,
        num_pics,
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
        real_pics = dataset.sample_observations(num_pics, random_state)

        result = batched_encoder(f, images=real_pics)

        means = result['mean']
        logvars = result['logvar']

        print(means.std(axis=0).round(3))
        print(np.exp(logvars).mean(axis=0).round(3))


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
    parser.add_argument('--num_pics', type=int, default=64)
    args = parser.parse_args()

    main(
        model_dir=args.model_dir,
        # output_dir=args.output_dir,
        num_pics=args.num_pics,
    )
