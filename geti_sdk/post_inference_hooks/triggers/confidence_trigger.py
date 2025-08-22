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


class ConfidenceTrigger(PostInferenceTrigger):
    """
    Post inference trigger based on the confidence of the predictions for an image.
    If the prediction confidence is **lower** than the defined `threshold`, the trigger
    is activated.

    Optionally a list of `label_names` can be defined, if passed then only the
    confidence level for the specified labels will be considered

    :param threshold: Confidence threshold to consider. Any predictions with a
        confidence **lower** than this threshold will activate the trigger
    :param label_names: Optional list of label names to include. If passed, only the
        confidences for the specified labels will be considered.
    """

    def __init__(self, threshold: float = 0.5, label_names: list[str] | None = None):
        super().__init__(threshold=threshold)
        self.label_names = label_names

        if label_names is not None:
            self._repr_info_ += f", label_names={label_names}"

    def __call__(self, image: np.ndarray, prediction: Prediction) -> float:
        """
        Compute a trigger score for the `image` and corresponding `prediction`.

        :param image: Numpy array representing an image
        :param prediction: Prediction object corresponding to the inference result
            for the image.
        :return: Float representing the score for the input
        """
        min_confidence: float = 1
        for predicted_object in prediction.annotations:
            for label in predicted_object.labels:
                if self.label_names is not None and label.name not in self.label_names:
                    continue
                if label.probability < min_confidence:
                    min_confidence = label.probability
        return min_confidence

    def get_decision(self, score: float) -> bool:
        """
        Make a decision based on a previously computed `score` and the threshold defined
        for the trigger

        :param score: Float representing the score for a certain image, prediction pair
        :return: Boolean indicating whether the trigger conditions are met (True,
            score is **lower** than the threshold), or not (False)
        """
        return score < self.threshold
