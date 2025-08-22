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


class EmptyLabelTrigger(PostInferenceTrigger):
    """
    Post inference trigger that will activate if the prediction is empty
    """

    def __init__(self):
        # LabelTrigger will return a score of 1 if label is found, so we can use the
        # default threshold defined in the super class
        super().__init__()
        self._repr_info_ = ""

    def __call__(self, image: np.ndarray, prediction: Prediction) -> float:
        """
        Compute a trigger score for the `image` and corresponding `prediction`.

        :param image: Numpy array representing an image
        :param prediction: Prediction object corresponding to the inference result
            for the image.
        :return: Float representing the score for the input
        """
        for predicted_object in prediction.annotations:
            if len(predicted_object.labels) > 1:
                return 0
            if not predicted_object.shape.is_full_box(image_width=image.shape[1], image_height=image.shape[0]):
                return 0
            for label in predicted_object.labels:
                if label.name.lower() == "no object":
                    return 1
        return 0
