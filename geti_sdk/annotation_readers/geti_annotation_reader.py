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

import glob
import json
import logging
import os
import warnings
from random import sample
from typing import Any

from geti_sdk.data_models import Annotation, TaskType
from geti_sdk.data_models.media import MediaInformation
from geti_sdk.data_models.shapes import Rectangle
from geti_sdk.rest_converters import AnnotationRESTConverter
from geti_sdk.rest_converters.annotation_rest_converter import (
    NormalizedAnnotationRESTConverter,
)

from .base_annotation_reader import AnnotationReader


class GetiAnnotationReader(AnnotationReader):
    """
    AnnotationReader for loading annotation files in Intel® Geti™ format.
    """

    def __init__(
        self,
        base_data_folder: str,
        annotation_format: str = ".json",
        task_type: TaskType | str | None = None,
        label_names_to_include: list[str] | None = None,
        anomaly_reduction: bool = False,
    ):
        """
        :param base_data_folder: Path to the folder containing the annotations
        :param annotation_format: Extension of the annotation files. Defaults to ".json"
        :param task_type: Optional task type to prepare the annotations for. Can also
            be specified later.
        :param label_names_to_include: Names of the labels that should be included
            when reading annotation data. This can be used to filter the annotations
            for certain labels.
        :param anomaly_reduction: True to reduce all anomaly tasks to the single anomaly task.
            This is done in accordance with the Intel Geti 2.5 Anomaly Reduction effort.
            All pixel level annotations are converted to full rectangles. All anomaly tasks
            are mapped to th new "Anomaly Detection" task wich corresponds to the old "Anomaly Classification".
        """
        if annotation_format != ".json":
            raise ValueError(
                f"Annotation format {annotation_format} is currently not supported by the GetiAnnotationReader"
            )
        super().__init__(
            base_data_folder=base_data_folder,
            annotation_format=annotation_format,
            task_type=task_type,
            anomaly_reduction=anomaly_reduction,
        )
        self._label_names_to_include = label_names_to_include
        self._normalized_annotations = self._has_normalized_annotations()

    def _get_label_names(self, all_labels: list[str]) -> list[str]:
        """
        Return the labels for the task type the annotation reader is currently set to.

        :param all_labels: List of all label names in the project
        """
        if self._label_names_to_include is None:
            logging.info("No label mapping defined, including all labels")
            labels = all_labels
        else:
            labels = [label for label in all_labels if label in self._label_names_to_include]
        return labels

    def _get_raw_annotation_data(self, filename: str) -> dict[str, Any]:
        """
        Read the annotation data from the file at `filename`

        :param filename: Name of the annotation file to read
        :return: Dictionary holding the annotation data
        """
        filepath = glob.glob(
            os.path.join(self.base_folder, f"{filename}{self.annotation_format}"),
        )
        if len(filepath) > 1:
            warnings.warn(
                f"Multiple matching annotation files found for image with name {filename}. Skipping this image..."
            )
            data = {"annotations": []}
        elif len(filepath) == 0:
            logging.info(f"No matching annotation file found for image with name {filename}. Skipping this image...")
            data = {"annotations": []}
        else:
            filepath = filepath[0]
            with open(filepath) as f:
                data = json.load(f)
        return data

    def get_data(
        self,
        filename: str,
        label_name_to_id_mapping: dict,
        media_information: MediaInformation,
        preserve_shape_for_global_labels: bool = False,
    ) -> list[Annotation]:
        """
        Return the annotation data for the dataset item corresponding to `filename`.

        :param filename: name of the item to get the annotation data for.
        :param label_name_to_id_mapping: mapping of label name to label id.
        :param media_information: MediaInformation object containing information
            (e.g. width, height) about the media item to upload the annotation for
        :param preserve_shape_for_global_labels: False to convert shapes for global
            tasks to full rectangles (required for classification like tasks in
            Intel® Geti™ projects), True to preserve such shapes. This parameter
            should be:

             - False when uploading annotations to a single task project
             - True when uploading annotations for a classification like task,
                following a local task in a task chain project.

        :return: List of Annotation objects containing all annotations for the given
            dataset item.
        """
        data = self._get_raw_annotation_data(filename=filename)

        new_annotations = []
        for annotation in data["annotations"]:
            if self._normalized_annotations:
                annotation_object = NormalizedAnnotationRESTConverter.normalized_annotation_from_dict(
                    annotation,
                    image_width=media_information.width,
                    image_height=media_information.height,
                )
            else:
                annotation_object = AnnotationRESTConverter.annotation_from_dict(annotation)
            for label in annotation_object.labels:
                label.id = label_name_to_id_mapping[label.name]
            for label_dict in annotation["labels"]:
                if self.task_type is not None and label_dict["name"] not in self._get_label_names(
                    list(label_name_to_id_mapping.keys())
                ):
                    annotation_object.pop_label_by_name(label_name=label_dict["name"])
            new_annotations.append(annotation_object)
            if self.anomaly_reduction and annotation_object.labels[0].name.lower() == "anomalous":
                # Part of anomaly task reduction in Intel Geti 2.5 -> all anomaly tasks combined into one.
                # Intel Geti now only accepts full rectangles for anomaly tasks.
                new_annotations = [
                    Annotation(
                        labels=[annotation_object.labels[0]],
                        shape=Rectangle.generate_full_box(
                            image_width=media_information.width,
                            image_height=media_information.height,
                        ),
                    )
                ]
                break
        return new_annotations

    def get_all_label_names(self) -> list[str]:
        """
        Retrieve the unique label names for all annotations in the annotation folder

        :return: List of label names
        """
        logging.info(f"Reading annotation files in folder {self.base_folder}...")
        unique_label_names = []
        annotation_files = glob.glob(
            os.path.join(self.base_folder, f"*{self.annotation_format}"),
        )
        if len(annotation_files) == 0:
            raise ValueError(f"No valid annotation files were found in folder {self.base_folder}")
        for annotation_file in annotation_files:
            with open(annotation_file) as f:
                data = json.load(f)
            annotations = data.get("annotations", None)
            if annotations is None:
                raise ValueError(
                    f"Annotation file '{annotation_file}' does not contain any "
                    f"annotations. Please make sure that this is a valid "
                    f"annotation file."
                )
            for annotation in annotations:
                labels = [label["name"] for label in annotation["labels"]]
                for label in labels:
                    if label not in unique_label_names:
                        unique_label_names.append(label)
        return unique_label_names

    def _has_normalized_annotations(self) -> bool:
        """
        Check if the annotation files belonging to this annotation reader are normalized
        """
        filenames = self.get_data_filenames()
        n_sample = min(len(filenames), 50)
        # nosec: is not used for security/cryptographic purposes
        list_to_check = sample(filenames, n_sample) if n_sample == 50 else filenames  # nosec B311

        NORMALIZED_KEY = "normalized"
        PIXEL_KEY = "pixel"

        annotation_stats = {NORMALIZED_KEY: 0, PIXEL_KEY: 0}
        for filename in list_to_check:
            data = self._get_raw_annotation_data(filename=filename)
            for annotation_dict in data["annotations"]:
                annotation_object = AnnotationRESTConverter.annotation_from_dict(annotation_dict)
                shape = annotation_object.shape
                x_max, y_max = shape.x_max, shape.y_max
                if x_max <= 1 and y_max <= 1:
                    annotation_stats[NORMALIZED_KEY] += 1
                else:
                    annotation_stats[PIXEL_KEY] += 1

        if annotation_stats[NORMALIZED_KEY] == 0:
            return False
        if annotation_stats[PIXEL_KEY] == 0:
            logging.info(
                "Legacy annotation format detected. The annotations you are trying to "
                "upload were most likely downloaded from a pre-production version of "
                "the Intel Geti software. They will be converted to the latest "
                "annotation format upon upload to the Intel Geti platform. "
            )
            return True
        raise ValueError(
            f"The annotation directory '{self.base_folder}' contains both "
            f"normalized ({annotation_stats[NORMALIZED_KEY]} shapes) and "
            f"non-normalized ({annotation_stats[PIXEL_KEY]} shapes) objects. "
            f"Unable to parse annotation data."
        )
