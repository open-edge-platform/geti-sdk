# Copyright (C) 2022 Intel Corporation
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

import os
import tempfile

import cv2
import numpy as np

from geti_sdk.data_models import Image, Prediction
from geti_sdk.utils import show_image_with_annotation_scene


def add_text_to_top_of_image(image: np.ndarray, text: str) -> np.ndarray:
    """
    Adds the specified text to the center top of the image.

    :param image: Image to add text to
    :param text: String to display
    :return: image with text added to the top center
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 1
    text_size, baseline = cv2.getTextSize(text, font, font_scale, thickness)
    origin = [int(0.5 * (image.shape[1] - text_size[0])), 2 + text_size[1]]
    cv2.putText(
        image,
        text,
        org=origin,
        fontFace=font,
        fontScale=font_scale,
        color=(0, 0, 0),
        thickness=thickness,
    )
    return image


def plot_predictions_side_by_side(
    image: Image | np.ndarray,
    prediction_1: Prediction,
    prediction_2: Prediction,
    filepath: str,
):
    """
    Plots two predictions next to each other

    :param image: Image on which to plot the predictions
    :param prediction_1: First Prediction for image
    :param prediction_2: Second Prediction for image
    :param filepath: Path to file in which to save the resulting image
    :return: resulting image
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        im_1_pred_path = os.path.join(temp_dir, "im_1.jpg")
        im_2_pred_path = os.path.join(temp_dir, "im_2.jpg")
        show_image_with_annotation_scene(image=image, annotation_scene=prediction_1, filepath=im_1_pred_path)
        show_image_with_annotation_scene(image=image, annotation_scene=prediction_2, filepath=im_2_pred_path)
        im1 = cv2.imread(im_1_pred_path)
        im2 = cv2.imread(im_2_pred_path)

    add_text_to_top_of_image(im1, "local prediction")
    add_text_to_top_of_image(im2, "online prediction")
    separator = np.zeros((im1.shape[0], 1, im1.shape[2]))
    final_im = np.concatenate((im1, separator, im2), axis=1)
    cv2.imwrite(filepath, final_im)
    return final_im
