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
import logging
import math
import time
from collections.abc import Sequence
from random import randint


def generate_segmentation_labels(detection_labels: list[str]) -> list[str]:
    """
    Generate segmentation label names from a list of detection label names.

    :param detection_labels: label names to generate segmentation label names for
    :return: List of label names
    """
    return [f"{label} shape" for label in detection_labels]


def generate_classification_labels(labels: list[str], multilabel: bool = False) -> list[dict[str, str]]:
    """
    Generate label creation data from a list of label names. If `multiclass = True`,
    the labels will be generated in such a way that multiple labels can be assigned to
    a single image. If `multiclass = False`, only a single label per image can be
    assigned.

    :param labels: Label names to be used
    :param multilabel: True to be able to assign multiple labels to one image, False
        otherwise. Defaults to False.
    :return: List of dictionaries containing the label data that can be sent to the
        Intel® Geti™ project creation endpoint
    """
    label_list: list[dict[str, str]] = []
    if multilabel or len(labels) == 1:
        for label in labels:
            label_list.append({"name": label, "group": f"{label}_group"})
    else:
        for label in labels:
            label_list.append({"name": label, "group": "default_classification_group"})
    return label_list


def generate_unique_label_color(label_colors: Sequence[str]) -> str:
    """
    Generate a label color that is unique from the label colors used in `labels`.

    :param label_colors: List of hex color strings with respect to which the new color
        should be generated
    :return: hex string containing the new label color
    """

    def _generate_random_rgb_tuple() -> tuple[int, int, int]:
        """
        Generate a random R,G,B color tuple. R,G,B are integers on the interval [0,255].

        :return: RGB tuple of integers in [0, 255]
        """
        return randint(0, 255), randint(0, 255), randint(0, 255)  # nosec B311

    def _calculate_rgb_distance(color_a: tuple[int, int, int], color_b: tuple[int, int, int]) -> float:
        """
        Calculate the root-mean-square difference between two RGB color tuples.
        """
        return math.sqrt(
            (color_a[0] - color_b[0]) ** 2 + (color_a[1] - color_b[1]) ** 2 + (color_a[2] - color_b[2]) ** 2
        )

    existing_colors = [tuple(int(label[i : i + 2], 16) for i in (1, 3, 5)) for label in label_colors]

    success = False
    t_start = time.time()
    new_color = _generate_random_rgb_tuple()
    distance_threshold = 30 if len(label_colors) < 100 else 10
    while not success and (time.time() - t_start < 100):
        new_color = _generate_random_rgb_tuple()
        min_distance = min([_calculate_rgb_distance(color, new_color) for color in existing_colors])
        if min_distance > distance_threshold:
            success = True
    if not success:
        logging.warning("Unable to generate sufficiently distinct label color.")
    return "#{:02x}{:02x}{:02x}".format(*new_color)
