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
import json
import logging
import os
from datetime import datetime

import cv2
import numpy as np

from geti_sdk.data_models import Prediction
from geti_sdk.deployment.inference_hook_interfaces import PostInferenceAction
from geti_sdk.prediction_visualization.visualizer import Visualizer
from geti_sdk.rest_converters import PredictionRESTConverter


class FileSystemDataCollection(PostInferenceAction):
    """
    Post inference action that will save an image to a specified folder on disk. The
    prediction output and trigger score that triggered the action are also saved.

    The data is saved in the `target_folder`, in which the action will create the
    following folder structure:

    <target_folder>
        |
        |- images
        |- predictions
        |- scores

    :param target_folder: Target folder on disk where the inferred images should be
        saved. If it does not exist yet, this action will create it.
    :param file_name_prefix: Prefix to use for the files that will be saved by this
        action. Default is 'image'
    :param save_predictions: True to save the predictions for each image as well, in
        addition to the image itself. Set to False to not store any predictions
    :param save_scores: True to save the trigger score for each image as well, in
        addition to the image itself. Set to False to not save any scores.
    :param log_level: Log level for the action. Options are 'info' or 'debug'
    """

    def __init__(
        self,
        target_folder: str,
        file_name_prefix: str = "",
        save_predictions: bool = True,
        save_scores: bool = True,
        save_overlays: bool = True,
        log_level: str = "debug",
    ):
        super().__init__(log_level=log_level)

        self.image_path = os.path.join(target_folder, "images")
        folders_to_create = [self.image_path]
        if save_predictions:
            self.predictions_path = os.path.join(target_folder, "predictions")
            folders_to_create.append(self.predictions_path)
        if save_scores:
            self.scores_path = os.path.join(target_folder, "scores")
            folders_to_create.append(self.scores_path)
        if save_overlays:
            self.overlays_path = os.path.join(target_folder, "overlays")
            folders_to_create.append(self.overlays_path)

        for path in folders_to_create:
            os.makedirs(path, exist_ok=True, mode=0o770)

        self.prefix = file_name_prefix
        self.save_predictions = save_predictions
        self.save_scores = save_scores
        self.save_overlays = save_overlays
        self.visualizer = Visualizer()

        self._repr_info_ = (
            f"target_folder=`{target_folder}`, "
            f"file_prefix={file_name_prefix}, "
            f"save_predictions={save_predictions}, "
            f"save_scores={save_scores}, "
            f"save_overlays={save_overlays}"
        )

    def __call__(
        self,
        image: np.ndarray,
        prediction: Prediction,
        score: float | None = None,
        name: str | None = None,
        timestamp: datetime | None = None,
    ):
        """
        Execute the action, save the given `image` to the predefined target folder.
        The `prediction` and `score` are also saved.

        :param image: Numpy array representing an image
        :param prediction: Prediction object which was generated for the image
        :param score: Optional score computed from a post inference trigger
        :param name: String containing the name of the image
        :param timestamp: Datetime object containing the timestamp belonging to the
            image
        """
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # we use cv2 to encode the numpy array as image, so it expects an
        # image in BGR format. However, `Deployment.infer` requires RGB format, so
        # we have to convert
        name = self.prefix if name is None else self.prefix + name
        if timestamp is None:
            timestamp = datetime.now()

        filename = name + "_" + timestamp.strftime("%Y%m%dT%H-%M-%S-%f")

        success = cv2.imwrite(os.path.join(self.image_path, filename + ".png"), image_bgr)
        if not success:
            logging.error(f"Failed to save image `{filename}.png` to folder `{self.image_path}`.")

        try:
            if self.save_predictions:
                prediction_filepath = os.path.join(self.predictions_path, filename + ".json")
                with open(prediction_filepath, "w") as file:
                    prediction_dict = PredictionRESTConverter.to_dict(prediction)
                    json.dump(prediction_dict, fp=file)

            if self.save_scores and score is not None:
                score_filepath = os.path.join(self.scores_path, filename + ".txt")
                with open(score_filepath, "w") as file:
                    file.write(f"{score}")

            if self.save_overlays:
                overlay_path = os.path.join(self.overlays_path, filename + ".jpg")
                result = self.visualizer.draw(image, prediction)
                self.visualizer.save_image(result, overlay_path)
        except Exception as e:
            logging.exception(e, stack_info=True, exc_info=True)

        self.log_function(f"FileSystemDataCollection inference action saved image data to folder `{self.image_path}`")
