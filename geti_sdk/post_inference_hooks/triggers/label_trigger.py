# Copyright (C) 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import numpy as np

from geti_sdk.data_models import Prediction
from geti_sdk.deployment.inference_hook_interfaces import PostInferenceTrigger


class LabelTrigger(PostInferenceTrigger):
    """
    Post inference trigger based on the presence of certain label(s) in the
    predictions for an image.
    If the specified label or labels are in the prediction results for an inferred
    image, the trigger is activated.

    :param label_names: List of label names to search for. If any of
    :param mode: Operation mode to use when evaluating the labels. Supported modes are:
        - `OR` (the default) -> If **any** of the labels in `label_names` if found in
            the prediction, the trigger activates.
        - `AND` -> If and only if **all** of the labels in `label_names` are
            predicted within the image, the trigger activates
    """

    def __init__(self, label_names: list[str], mode: str = "OR"):
        self.label_names = set(label_names)
        self.mode = mode

        # LabelTrigger will return a score of 1 if label is found, so we can use the
        # default threshold defined in the super class
        super().__init__()
        self._repr_info_ = f"label_names={label_names}"
        self._repr_info_ += f", mode={mode}"

    def __call__(self, image: np.ndarray, prediction: Prediction) -> float:
        """
        Compute a trigger score for the `image` and corresponding `prediction`.

        :param image: Numpy array representing an image
        :param prediction: Prediction object corresponding to the inference result
            for the image.
        :return: Float representing the score for the input
        """
        predicted_labels = set()
        for label in prediction.get_labels():
            predicted_labels.add(label.name)
        if self.mode == "AND":
            return float(self.label_names.issubset(predicted_labels))
        # mode == "OR"
        return float(not self.label_names.isdisjoint(predicted_labels))
