import pandas as pd

DLIB_RESULTS_PATH = 'aggregated_results/dlib_results.json'

dlib_df = pd.read_json(DLIB_RESULTS_PATH)
beta_vae_df = dlib_df.loc[
    (dlib_df['train_config.model.name'] == "'beta_vae'")
    # & (dlib_df['train_config.dataset.name'] == "'dsprites_full'")
    ]
beta_vae_df.to_json(DLIB_RESULTS_PATH.replace('dlib_results', 'dlib_beta_vae_results'))
