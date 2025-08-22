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

SUPPORTED_MODES = ["greater", "lower", "equal"]


class ObjectCountTrigger(PostInferenceTrigger):
    """
    Post inference trigger based on the number of predicted objects in an image.

    The operation mode of the trigger can be specified: It will activate either if the
    object count is above the threshold, below the threshold or exactly equal to the
    threshold value. This behaviour can be controlled via the `mode` parameter.

    :param threshold: Threshold value for the total number of objects in a prediction
    :param label_names: Optional list of classes/labels to account for in the object
        counting. If specified, only objects with those labels will count towards the
        total
    :param mode: Operation mode for the trigger, which defines when the trigger will
        activate. Options are:

            - `greater`  -> Trigger activates if object count is greater than threshold
            - `lower`    -> Trigger activates if object count is lower than threshold
            - `equal`    -> Trigger activates if object count is equal to threshold

        Defaults to `greater`
    """

    def __init__(
        self,
        threshold: int = 1,
        label_names: list[str] | None = None,
        mode: str = "greater",
    ):
        super().__init__(threshold=threshold)
        self.label_names = label_names
        self.filter_labels = False

        if label_names is not None:
            self.filter_labels = True
            self._repr_info_ += f", label_names={label_names}"

        lower_mode = mode.lower()
        if lower_mode not in SUPPORTED_MODES:
            raise ValueError(f"Invalid mode `{mode}`. Valid options are: {SUPPORTED_MODES}")
        self.mode = lower_mode
        self._repr_info_ += f", mode={lower_mode}"

    def __call__(self, image: np.ndarray, prediction: Prediction) -> float:
        """
        Compute a trigger score for the `image` and corresponding `prediction`.

        :param image: Numpy array representing an image
        :param prediction: Prediction object corresponding to the inference result
            for the image.
        :return: Float representing the score for the input
        """
        if not self.filter_labels:
            n_objects = len(prediction.annotations)
            if n_objects != 1:
                return n_objects
            # Prediction might be a 'No object', we have to check for this
            predicted_labels = prediction.annotations[0].labels
            if len(predicted_labels) > 1:
                return 1.0
            if predicted_labels[0].name.lower() == "no object":
                return 0.0
            return 1.0
        object_count: int = 0
        for predicted_object in prediction.annotations:
            for label in predicted_object.labels:
                if label.name in self.label_names:
                    object_count += 1
        return object_count

    def get_decision(self, score: float) -> bool:
        """
        Make a decision based on a previously computed `score` and the threshold defined
        for the trigger

        :param score: Float representing the score for a certain image, prediction pair
        :return: Boolean indicating whether the trigger conditions are met (True), or
            not (False)
        """
        if self.mode == "lower":
            return score < self.threshold
        if self.mode == "greater":
            return score > self.threshold
        if self.mode == "equal":
            return score == self.threshold
        return None
