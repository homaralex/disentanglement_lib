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

from disentanglement_lib.config.sparsity_study import sweep as sparsity_study
from disentanglement_lib.config.abstract_reasoning_study_v1.stage1 import sweep as abstract_reasoning_study_v1
from disentanglement_lib.config.fairness_study_v1 import sweep as fairness_study_v1
from disentanglement_lib.config.tests import sweep as tests
from disentanglement_lib.config.unsupervised_study_v1 import sweep as unsupervised_study_v1
import disentanglement_lib.utils.hyperparams as h

_betas = h.sweep('beta', (1, 2, 4, 6, 8, 16))
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

_sweep_dim_wise = h.product((_betas, _datasets, _all_layers))
_masked_studies = {
    f"{s['dataset']}_masked_{'all_' if s['all_layers'] else ''}b_{s['beta']}": sparsity_study.MaskedSparsityStudy(**s)
    for s in _sweep_dim_wise
}

STUDIES = {
    "unsupervised_study_v1": unsupervised_study_v1.UnsupervisedStudyV1(),
    "abstract_reasoning_study_v1":
        abstract_reasoning_study_v1.AbstractReasoningStudyV1(),
    "fairness_study_v1":
        fairness_study_v1.FairnessStudyV1(),
    "test": tests.TestStudy(),
    'wae': sparsity_study.WAEStudy(dataset='dsprites_full'),
    **_dim_wise_studies,
    **_dim_wise_mask_studies,
    **_masked_studies,
}
