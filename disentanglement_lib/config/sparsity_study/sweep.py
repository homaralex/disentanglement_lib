from disentanglement_lib.config import study
from disentanglement_lib.utils import resources
import disentanglement_lib.utils.hyperparams as h
from six.moves import range

import numpy as np


class BaseSparsityStudy(study.Study):
    def __init__(self, beta, dataset):
        self.beta = beta
        self.dataset = dataset

    def get_datasets(self):
        return h.fixed('dataset.name', self.dataset)

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
            # TODO
            self.get_seeds(5),
            self.get_datasets(),
            architecture,
            self.get_default_models(),
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
    def __init__(
            self,
            dim='col',
            all_layers=True,
            scale_per_layer=False,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.dim = dim
        self.all_layers = all_layers
        self.scale_per_layer = scale_per_layer

    def get_default_models(self):
        # DimWiseL1 config.
        model_name = h.fixed("model.name", f"dim_wise_l1_{self.dim}_vae")
        model_fn = h.fixed("model.model", "@dim_wise_l1_vae()")
        beta = h.fixed('vae.beta', self.beta)
        scale_per_layer = h.fixed('dim_wise_l1_vae.scale_per_layer', self.scale_per_layer)
        lmbds_l1 = h.sweep("dim_wise_l1_vae.lmbd_l1", h.discrete([
            *np.logspace(-5, -3, 8)
        ]))
        dim = h.fixed('dim_wise_l1_vae.dim', self.dim)
        all_layers = h.fixed('dim_wise_l1_vae.all_layers', self.all_layers)
        config_dim_wise_l1 = h.zipit([
            model_name,
            model_fn,
            beta,
            scale_per_layer,
            lmbds_l1,
            dim,
            all_layers,
        ])

        all_models = h.chainit([config_dim_wise_l1, ])

        return all_models


class MaskedSparsityStudy(BaseSparsityStudy):
    def __init__(self, all_layers=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.all_layers = all_layers

    def get_default_models(self):
        model_name = h.fixed("model.name", "masked_vae")
        model_fn = h.fixed("model.model", "@vae()")
        beta = h.fixed('vae.beta', self.beta)
        perc_sparse = h.sweep("conv_encoder.perc_sparse", h.discrete([
            *np.linspace(.1, 1, 6, endpoint=False)
        ]))
        all_layers = h.fixed('conv_encoder.all_layers', self.all_layers)
        config_masked = h.zipit([
            model_name,
            model_fn,
            beta,
            perc_sparse,
            all_layers,
        ])

        all_models = h.chainit([config_masked, ])

        return all_models


class DimWiseMaskL1Study(DimWiseL1SparsityStudy):
    def __init__(self, lmbd_l1_range=np.logspace(-5, -3, 8), *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.lmbd_l1_range = lmbd_l1_range

    def get_default_models(self):
        model_name = h.fixed("model.name", "dim_wise_mask_l1_vae")
        model_fn = h.fixed("model.model", "@dim_wise_mask_l1_vae()")
        beta = h.fixed('vae.beta', self.beta)
        scale_per_layer = h.fixed('dim_wise_l1_vae.scale_per_layer', self.scale_per_layer)
        dim = h.fixed('dim_wise_l1_vae.dim', self.dim)
        all_layers_1 = h.fixed('dim_wise_l1_vae.all_layers', self.all_layers)
        all_layers_2 = h.fixed('conv_encoder.all_layers', self.all_layers)
        # make the masks full (no zero-entries)
        perc_sparse = h.fixed("conv_encoder.perc_sparse", h.discrete(0))
        # but allow to modify the entries
        mask_trainable = h.fixed('masked_layer.mask_trainable', True)
        lmbds_l2 = h.fixed('dim_wise_mask_l1_vae.lmbd_l2', h.discrete(.01))
        lmbds_l1 = h.sweep("dim_wise_l1_vae.lmbd_l1", h.discrete(self.lmbd_l1_range))
        config_masked_l1 = h.zipit([
            model_name,
            model_fn,
            beta,
            scale_per_layer,
            dim,
            all_layers_1,
            all_layers_2,
            perc_sparse,
            mask_trainable,
            lmbds_l2,
            lmbds_l1,
        ])

        all_models = h.chainit([config_masked_l1, ])

        return all_models


class WeigthDecaystudy(DimWiseMaskL1Study):
    def __init__(self, lmbd_l2_range=(.01,), *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.lmbd_l1_range = [0.]
        self.lmbd_l2_range = lmbd_l2_range

    def get_default_models(self):
        model_name = h.fixed("model.name", "weight_decay_vae")
        model_fn = h.fixed("model.model", "@dim_wise_mask_l1_vae()")
        beta = h.fixed('vae.beta', self.beta)
        scale_per_layer = h.fixed('dim_wise_l1_vae.scale_per_layer', self.scale_per_layer)
        dim = h.fixed('dim_wise_l1_vae.dim', self.dim)
        all_layers_1 = h.fixed('dim_wise_l1_vae.all_layers', self.all_layers)
        all_layers_2 = h.fixed('conv_encoder.all_layers', self.all_layers)
        # make the masks full (no zero-entries)
        perc_sparse = h.fixed("conv_encoder.perc_sparse", h.discrete(0))
        # but allow to modify the entries
        mask_trainable = h.fixed('masked_layer.mask_trainable', False)
        lmbds_l2 = h.sweep('dim_wise_mask_l1_vae.lmbd_l2', h.discrete(self.lmbd_l2_range))
        lmbds_l1 = h.fixed("dim_wise_l1_vae.lmbd_l1", h.discrete(self.lmbd_l1_range[0]))
        config_weight_decay = h.zipit([
            model_name,
            model_fn,
            beta,
            scale_per_layer,
            dim,
            all_layers_1,
            all_layers_2,
            perc_sparse,
            mask_trainable,
            lmbds_l2,
            lmbds_l1,
        ])

        all_models = h.chainit([config_weight_decay, ])

        return all_models


class MaskL1Study(DimWiseMaskL1Study):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dim = None


class WAEStudy(BaseSparsityStudy):
    def __init__(self, *args, **kwargs):
        # add a placeholder for beta
        super().__init__(beta=1, *args, **kwargs)

    def get_default_models(self):
        model_name = h.fixed("model.name", "wae")
        model_fn = h.fixed("model.model", "@wae()")
        scale = h.fixed("wae.scale", 1 / 8)
        adaptive = h.fixed("wae.adaptive", True)
        betas = h.sweep("vae.beta", h.discrete([*np.logspace(-4, -2, 3)]))
        config_vae = h.zipit([
            model_name,
            model_fn,
            scale,
            adaptive,
            betas,
        ])

        all_models = h.chainit([config_vae, ])

        return all_models
