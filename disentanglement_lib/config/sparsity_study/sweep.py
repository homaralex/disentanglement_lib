from disentanglement_lib.config import study
from disentanglement_lib.utils import resources
import disentanglement_lib.utils.hyperparams as h
from six.moves import range

import numpy as np


class BaseSparsityStudy(study.Study):
    def get_datasets(self):
        return h.sweep(
            "dataset.name",
            h.categorical([
                # TODO
                "dsprites_full",
                # "color_dsprites",
                # "noisy_dsprites",
                # "scream_dsprites",
                # "smallnorb",
                # "cars3d",
                # "shapes3d",
            ]))

    def get_num_latent(self, sweep):
        return h.sweep("encoder.num_latent", h.discrete(sweep))

    def get_seeds(self, num):
        """Returns random seeds."""
        return h.sweep("model.random_seed", h.categorical(list(range(num))))

    def get_config(self):
        """Returns the hyperparameter configs for different experiments."""
        arch_enc = h.fixed("encoder.encoder_fn", "@conv_encoder", length=1)
        arch_dec = h.fixed("decoder.decoder_fn", "@deconv_decoder", length=1)
        architecture = h.zipit([arch_enc, arch_dec])
        return h.product([
            self.get_datasets(),
            architecture,
            self.get_default_models(),
            # TODO
            self.get_seeds(5),
        ])

    def get_model_config(self, model_num=0):
        """Returns model bindings and config file."""
        config = self.get_config()[model_num]
        model_bindings = h.to_bindings(config)
        model_config_file = resources.get_file("config/unsupervised_study_v1/model_configs/shared.gin")
        return model_bindings, model_config_file

    def get_postprocess_config_files(self):
        """Returns postprocessing config files."""
        return [resources.get_file("config/unsupervised_study_v1/postprocess_configs/mean.gin")]

    def get_eval_config_files(self):
        """Returns evaluation config files."""
        return list(resources.get_files_in_folder("config/unsupervised_study_v1/metric_configs/"))


class BaselineSparsityStudy(BaseSparsityStudy):
    def get_default_models(self):
        # Baseline VAE
        model_name = h.fixed("model.name", "beta_vae")
        model_fn = h.fixed("model.model", "@vae()")
        # betas = h.sweep("vae.beta", h.discrete([1., 2., 4., 6., 8., 16.]))
        config_vae = h.zipit([
            model_name,
            # betas,
            model_fn,
        ])

        all_models = h.chainit([config_vae, ])

        return all_models


class DimWiseL1SparsityStudy(BaseSparsityStudy):
    def __init__(self, dim='col'):
        self.dim = dim

    def get_default_models(self):
        # DimWiseL1 config.
        model_name = h.fixed("model.name", f"dim_wise_l1_{self.dim}_vae")
        model_fn = h.fixed("model.model", "@dim_wise_l1_vae()")
        # betas = h.sweep("vae.beta", h.discrete([1., 2., 4., 6., 8., 16.]))
        lmbds_l1 = h.sweep("dim_wise_l1_vae.lmbd_l1", h.discrete([
            *np.logspace(-5, 0, 6)
        ]))
        dim = h.fixed('dim_wise_l1_vae.dim', self.dim)
        all_layers = h.fixed('dim_wise_l1_vae.all_layers', True)
        config_dim_wise_l1 = h.zipit([
            model_name,
            model_fn,
            lmbds_l1,
            dim,
            all_layers,
        ])

        all_models = h.chainit([config_dim_wise_l1, ])

        return all_models
