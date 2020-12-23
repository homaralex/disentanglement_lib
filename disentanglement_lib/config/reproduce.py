# coding=utf-8
# Copyright 2018 The DisentanglementLib Authors.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Different studies that can be reproduced."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from disentanglement_lib.config.sparsity_study import sweep as sparsity_study
from disentanglement_lib.config.greedy_study import sweep as greedy_study
from disentanglement_lib.config.abstract_reasoning_study_v1.stage1 import sweep as abstract_reasoning_study_v1
from disentanglement_lib.config.fairness_study_v1 import sweep as fairness_study_v1
from disentanglement_lib.config.tests import sweep as tests
from disentanglement_lib.config.unsupervised_study_v1 import sweep as unsupervised_study_v1
import disentanglement_lib.utils.hyperparams as h

_betas = h.sweep('beta', (1, 2, 4, 6, 8, 16, 32, 64))
_datasets = h.sweep('dataset', (
    "dsprites_full",
    "color_dsprites",
    "noisy_dsprites",
    "scream_dsprites",
    "smallnorb",
    "cars3d",
    "shapes3d",
))
_all_layers = h.sweep('all_layers', (True, False))
_scale_per_layer = h.sweep('scale_per_layer', (True, False))
_dims = h.sweep('dim', ('col', 'row'))
_sweep_dim_wise = h.product((_betas, _datasets, _all_layers, _scale_per_layer, _dims))

_dim_wise_studies = {
    f"{s['dataset']}_dim_wise_l1_{s['dim']}_{'all_' if s['all_layers'] else ''}{'scale_' if s['scale_per_layer'] else ''}b_{s['beta']}": sparsity_study.DimWiseL1SparsityStudy(
        **s)
    for s in _sweep_dim_wise
}
_dim_wise_mask_studies = {
    f"{s['dataset']}_dim_wise_mask_l1_{s['dim']}_{'all_' if s['all_layers'] else ''}{'scale_' if s['scale_per_layer'] else ''}b_{s['beta']}": sparsity_study.DimWiseMaskL1Study(
        **s)
    for s in _sweep_dim_wise
}
_dim_wise_mask_studies_2 = {
    f"{s['dataset']}_dim_wise_mask_2_l1_{s['dim']}_{'all_' if s['all_layers'] else ''}{'scale_' if s['scale_per_layer'] else ''}b_{s['beta']}": sparsity_study.DimWiseMaskL1Study(
        lmbd_l1_range=np.logspace(-7, -5, 3, endpoint=False),
        **s)
    for s in _sweep_dim_wise
}
_dim_wise_mask_studies_3 = {
    f"{s['dataset']}_dim_wise_mask_3_l1_{s['dim']}_{'all_' if s['all_layers'] else ''}{'scale_' if s['scale_per_layer'] else ''}b_{s['beta']}": sparsity_study.DimWiseMaskL1Study(
        lmbd_l1_range=np.logspace(-4, -2, 5),
        **s)
    for s in _sweep_dim_wise
}
_dim_wise_mask_studies_4 = {
    f"{s['dataset']}_dim_wise_mask_4_l1_{s['dim']}_{'all_' if s['all_layers'] else ''}{'scale_' if s['scale_per_layer'] else ''}b_{s['beta']}": sparsity_study.DimWiseMaskL1Study(
        lmbd_l1_range=np.logspace(0, -2, 4, endpoint=False),
        **s)
    for s in _sweep_dim_wise
}
_dim_wise_mask_studies_5 = {
    f"{s['dataset']}_dim_wise_mask_5_l1_{s['dim']}_{'all_' if s['all_layers'] else ''}{'scale_' if s['scale_per_layer'] else ''}b_{s['beta']}": sparsity_study.DimWiseMaskL1Study(
        lmbd_l1_range=np.logspace(-9, -7, 4, endpoint=False),
        **s)
    for s in _sweep_dim_wise
}
_dim_wise_mask_studies_6 = {
    f"{s['dataset']}_dim_wise_mask_6_l1_{s['dim']}_{'all_' if s['all_layers'] else ''}{'scale_' if s['scale_per_layer'] else ''}b_{s['beta']}": sparsity_study.DimWiseMaskL1Study(
        lmbd_l1_range=np.logspace(-6, -4, 4, endpoint=False),
        **s)
    for s in _sweep_dim_wise
}
_dim_wise_mask_studies_7 = {
    f"{s['dataset']}_dim_wise_mask_7_l1_{s['dim']}_{'all_' if s['all_layers'] else ''}{'scale_' if s['scale_per_layer'] else ''}b_{s['beta']}": sparsity_study.DimWiseMaskL1Study(
        lmbd_l1_range=(.1,),
        **s)
    for s in _sweep_dim_wise
}
_mask_l1_studies = {
    f"{s['dataset']}_mask_l1_{'all_' if s['all_layers'] else ''}{'scale_' if s['scale_per_layer'] else ''}b_{s['beta']}": sparsity_study.MaskL1Study(
        lmbd_l1_range=np.logspace(-8, -3, 6),
        **s)
    for s in _sweep_dim_wise
}
_mask_l1_studies_2 = {
    f"{s['dataset']}_mask_2_l1_{'all_' if s['all_layers'] else ''}{'scale_' if s['scale_per_layer'] else ''}b_{s['beta']}": sparsity_study.MaskL1Study(
        lmbd_l1_range=np.logspace(-4, -2, 5),
        **s)
    for s in _sweep_dim_wise
}
_mask_l1_studies_3 = {
    f"{s['dataset']}_mask_3_l1_{'all_' if s['all_layers'] else ''}{'scale_' if s['scale_per_layer'] else ''}b_{s['beta']}": sparsity_study.MaskL1Study(
        lmbd_l1_range=np.logspace(0, -2, 4, endpoint=False),
        **s)
    for s in _sweep_dim_wise
}
_mask_l1_studies_4 = {
    f"{s['dataset']}_mask_4_l1_{'all_' if s['all_layers'] else ''}{'scale_' if s['scale_per_layer'] else ''}b_{s['beta']}": sparsity_study.MaskL1Study(
        lmbd_l1_range=np.logspace(-10, -8, 4, endpoint=False),
        **s)
    for s in _sweep_dim_wise
}
_mask_l1_studies_5 = {
    f"{s['dataset']}_mask_5_l1_{'all_' if s['all_layers'] else ''}{'scale_' if s['scale_per_layer'] else ''}b_{s['beta']}": sparsity_study.MaskL1Study(
        lmbd_l1_range=np.logspace(-6, -4, 4, endpoint=False),
        **s)
    for s in _sweep_dim_wise
}
_mask_l1_studies_7 = {
    f"{s['dataset']}_mask_7_l1_{'all_' if s['all_layers'] else ''}{'scale_' if s['scale_per_layer'] else ''}b_{s['beta']}": sparsity_study.MaskL1Study(
        lmbd_l1_range=(.1,),
        **s)
    for s in _sweep_dim_wise
}

_mask_l1_studies_paper = {
    f"{s['dataset']}_mask_paper_l1_{'all_' if s['all_layers'] else ''}{'scale_' if s['scale_per_layer'] else ''}b_{s['beta']}": sparsity_study.MaskL1Study(
        lmbd_l1_range=np.logspace(-1, -6, 6),
        **s)
    for s in _sweep_dim_wise
}

_weight_decay_studies = {
    f"{s['dataset']}_weight_decay_{'all_' if s['all_layers'] else ''}b_{s['beta']}": sparsity_study.WeigthDecayStudy(
        lmbd_l2_range=(.01,),
        **s)
    for s in _sweep_dim_wise
}

_sweep_masked = h.product((_betas, _datasets, _all_layers))
_masked_studies = {
    f"{s['dataset']}_masked_{'all_' if s['all_layers'] else ''}b_{s['beta']}": sparsity_study.MaskedSparsityStudy(**s)
    for s in _sweep_masked
}

_sweep_proximal = h.product((_betas, _datasets, _all_layers))
_proximal_studies = {
    f"{s['dataset']}_proximal_{'all_' if s['all_layers'] else ''}b_{s['beta']}": sparsity_study.ProximalStudy(**s)
    for s in _sweep_proximal
}

_anneal = h.sweep('anneal_kld_for', (None, 100000))
_sweep_variational = h.product((_betas, _datasets, _all_layers, _anneal, _scale_per_layer))
_variational_studies = {
    f"{s['dataset']}_vd_{'anneal_' if s['anneal_kld_for'] is not None else ''}{'all_' if s['all_layers'] else ''}{'scale_' if s['scale_per_layer'] else ''}b_{s['beta']}": sparsity_study.VDStudy(
        **s)
    for s in _sweep_variational
}

_scale_temperature = h.sweep('scale_temperature', (True, False))
_sweep_softmax = h.product((_betas, _datasets, _all_layers, _scale_temperature))
_softmax_studies = {
    f"{s['dataset']}_softmax_{'all_' if s['all_layers'] else ''}{'scale_' if s['scale_temperature'] else ''}b_{s['beta']}": sparsity_study.SoftmaxStudy(
        **s)
    for s in _sweep_softmax
}

_sweep_small = h.product((_betas, _datasets))
_small_studies = {
    f"{s['dataset']}_small_vae_b_{s['beta']}": sparsity_study.SmallVAEStudy(**s)
    for s in _sweep_small
}

_sweep_baseline = h.product((_betas, _datasets))
_baseline_studies = {
    f"{s['dataset']}_baseline_b_{s['beta']}": sparsity_study.BaselineSparsityStudy(**s)
    for s in _sweep_baseline
}

_sweep_dropout = h.product((_betas, _datasets, _all_layers))
_dropout_studies = {
    f"{s['dataset']}_dropout_{'all_' if s['all_layers'] else ''}b_{s['beta']}": sparsity_study.DropoutStudy(**s)
    for s in _sweep_dropout
}

_sweep_greedy = h.product((_betas, _datasets))
_greedy_studies = {
    f"{s['dataset']}_greedy_b_{s['beta']}": sparsity_study.GreedyStudy(**s)
    for s in _sweep_greedy
}

_balanced_weighing = h.sweep('balanced', (True, False))
_sweep_hnlpca = h.product((_betas, _datasets, _balanced_weighing))
_hnlpca_studies = {
    f"{s['dataset']}_hnlpca{'_balanced' if s['balanced'] else ''}_b_{s['beta']}": greedy_study.HNLPCAStudy(**s)
    for s in _sweep_hnlpca
}

_code_norm = h.sweep('code_norm', (True, False))
_sweep_wae = h.product((_datasets, _code_norm))
_wae_studies = {
    f"{s['dataset']}_wae{'_norm' if s['code_norm'] else ''}": sparsity_study.WAEStudy(**s)
    for s in _sweep_wae
}

STUDIES = {
    "unsupervised_study_v1": unsupervised_study_v1.UnsupervisedStudyV1(),
    "abstract_reasoning_study_v1":
        abstract_reasoning_study_v1.AbstractReasoningStudyV1(),
    "fairness_study_v1":
        fairness_study_v1.FairnessStudyV1(),
    "test": tests.TestStudy(),
    **_dim_wise_studies,
    **_dim_wise_mask_studies,
    **_dim_wise_mask_studies_2,
    **_dim_wise_mask_studies_3,
    **_dim_wise_mask_studies_4,
    **_dim_wise_mask_studies_5,
    **_dim_wise_mask_studies_6,
    **_dim_wise_mask_studies_7,
    **_mask_l1_studies,
    **_mask_l1_studies_2,
    **_mask_l1_studies_3,
    **_mask_l1_studies_4,
    **_mask_l1_studies_5,
    **_mask_l1_studies_7,

    **_mask_l1_studies_paper,
    **_masked_studies,
    **_proximal_studies,
    **_variational_studies,
    **_softmax_studies,
    **_weight_decay_studies,
    **_small_studies,
    **_dropout_studies,
    **_greedy_studies,
    **_hnlpca_studies,

    **_baseline_studies,

    **_wae_studies,
}
