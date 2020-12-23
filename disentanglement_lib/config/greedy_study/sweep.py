from disentanglement_lib.config.sparsity_study.sweep import BaseSparsityStudy
import disentanglement_lib.utils.hyperparams as h


class HNLPCAStudy(BaseSparsityStudy):
    def __init__(
            self,
            balanced=False,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.balanced = balanced

    def get_default_models(self):
        model_name = h.fixed("model.name", "hnlpca")
        model_fn = h.fixed("model.model", "@hlnpca()")
        beta = h.fixed('vae.beta', self.beta)
        balanced = h.fixed('hlnpca.balanced', self.balanced)

        config_hnlpca = h.zipit([
            model_name,
            beta,
            balanced,
            model_fn,
        ])

        all_models = h.chainit([config_hnlpca, ])

        return all_models
