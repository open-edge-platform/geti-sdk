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

import copy
import logging
from typing import Dict, List, Optional, Sequence, Union

import cv2
import numpy as np
from datumaro import Image
from datumaro.components.annotation import Bbox, Points, Polygon

from geti_sdk.annotation_readers.base_annotation_reader import AnnotationReader
from geti_sdk.data_models import Annotation as SCAnnotation
from geti_sdk.data_models import TaskType
from geti_sdk.data_models.enums.task_type import GLOBAL_TASK_TYPES
from geti_sdk.data_models.media import MediaInformation
from geti_sdk.rest_converters import AnnotationRESTConverter
from geti_sdk.utils import (
    deprecate,
    generate_segmentation_labels,
    get_dict_key_from_value,
)

from .datumaro_dataset import DatumaroDataset


@deprecate(
    since="2.12",
    use="GetiIE",
    reason="Dataset and annotations can be imported using Geti's native Dataset Import/Export functionality.",
)
class DatumAnnotationReader(AnnotationReader):
    """
    Class to read annotations using datumaro
    """

    _SUPPORTED_TASK_TYPES = [
        TaskType.DETECTION,
        TaskType.SEGMENTATION,
        TaskType.CLASSIFICATION,
        TaskType.INSTANCE_SEGMENTATION,
        TaskType.ROTATED_DETECTION,
        TaskType.ANOMALY_CLASSIFICATION,
        TaskType.ANOMALY_DETECTION,
        TaskType.ANOMALY_SEGMENTATION,
        TaskType.ANOMALY,
        TaskType.KEYPOINT_DETECTION,
    ]

    def __init__(
        self,
        base_data_folder: str,
        annotation_format: str,
        task_type: Union[TaskType, str] = TaskType.DETECTION,
    ):
        super().__init__(
            base_data_folder=base_data_folder,
            annotation_format=annotation_format,
            task_type=task_type,
        )
        self.dataset = DatumaroDataset(
            dataset_format=annotation_format, dataset_path=base_data_folder
        )
        self._override_label_map: Optional[Dict[int, str]] = None
        self._applied_filters: List[Dict[str, Union[List[str], str]]] = []

    def prepare_and_set_dataset(
        self,
        task_type: Union[TaskType, str],
        previous_task_type: Optional[TaskType] = None,
    ) -> None:
        """
        Prepare the dataset for a specific `task_type`. This could involve for
        example conversion of annotation shapes.

        :param task_type: TaskType for which to prepare the dataset.
        :param previous_task_type: TaskType preceding the task to prepare the dataset
            for
        """
        if not isinstance(task_type, TaskType):
            task_type = TaskType(task_type)
        if task_type != self.task_type:
            logging.info(f"Task type changed to {task_type} for dataset")
            if task_type not in self._SUPPORTED_TASK_TYPES:
                raise ValueError(f"Unsupported task type {task_type}")
            new_dataset = DatumaroDataset(
                dataset_format=self.annotation_format, dataset_path=self.base_folder
            )
            self.task_type = task_type
            self.dataset = new_dataset
            for filter_parameters in self.applied_filters:
                self.filter_dataset(**filter_parameters)

        dataset = self.dataset.prepare_dataset(
            task_type=task_type, previous_task_type=previous_task_type
        )
        self.dataset.set_dataset(dataset)
        logging.info(f"Dataset is prepared for {task_type} task.")

    def convert_labels_to_segmentation_names(self) -> None:
        """
        Convert the label names in a dataset to '*_shape`, where `*` is
        the original label name. It can be used to generate unique label names for the
        segmentation task in a detection_to_segmentation project
        """
        segmentation_label_map: Dict[int, str] = {}
        label_names = list(self.datum_label_map.values())
        segmentation_label_names = generate_segmentation_labels(label_names)
        for datum_index, label_name in self.datum_label_map.items():
            label_index = label_names.index(label_name)
            segmentation_label_map.update(
                {datum_index: segmentation_label_names[label_index]}
            )
        self.override_label_map(segmentation_label_map)

    def get_all_label_names(self) -> List[str]:
        """
        Retrieve the list of labels names from a datumaro dataset.
        """
        return list(set(self.datum_label_map.values()))

    def get_keypoint_joints(self) -> set[tuple[int, int]]:
        """
        Return a list of tuples representing the joints between keypoints
        """
        return self.dataset.joints_mapping

    @property
    def datum_label_map(self) -> Dict[int, str]:
        """
        :return: Dictionary mapping the datumaro label id to the label name
        """
        if self._override_label_map is None:
            return self.dataset.label_mapping
        else:
            return self._override_label_map

    def override_label_map(self, new_label_map: Dict[int, str]):
        """
        Override the label map defined in the datumaro dataset
        """
        self._override_label_map = new_label_map

    def reset_label_map(self):
        """
        Reset the label map back to the original one from the datumaro dataset.
        """
        self._override_label_map = None

    def get_all_image_names(self) -> List[str]:
        """
        Return a list of image names in the dataset
        """
        return self.dataset.image_names

    def get_data(
        self,
        filename: str,
        label_name_to_id_mapping: dict,
        media_information: MediaInformation,
        preserve_shape_for_global_labels: bool = False,
    ) -> List[SCAnnotation]:
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
        ds_item = self.dataset.get_item_by_id(filename)
        image_size = ds_item.media_as(Image).size
        annotation_list: List[SCAnnotation] = []
        labels = []

        # Remove duplicate annotations, datumaro does not check for this
        datum_annotations = [
            i
            for n, i in enumerate(ds_item.annotations)
            if i not in ds_item.annotations[:n]
        ]

        for annotation in datum_annotations:
            try:
                label_name = self.datum_label_map[annotation.label]
            except KeyError:
                # Label is not in the Intel® Geti™ project labels, move on to next
                # annotation for this dataset item.
                continue

            if self.dataset.points_categories:
                # keypoint detection annotations are n:1 relation for Datumaro:Geti
                if isinstance(annotation, Points):
                    keypoint_names = self.dataset.points_names
                    for i in range(0, len(annotation.points), 2):
                        label_name = keypoint_names[i // 2]
                        label_id = label_name_to_id_mapping.get(label_name)
                        label = {"id": label_id, "probability": 1.0}
                        shape = {
                            "type": "KEYPOINT",
                            "x": float(annotation.points[i]),
                            "y": float(annotation.points[i + 1]),
                            "is_visible": (
                                True
                                if annotation.visibility[i // 2]
                                == Points.Visibility.visible
                                else False
                            ),
                        }
                        sc_annotation = AnnotationRESTConverter.annotation_from_dict(
                            {"labels": [label], "shape": shape}
                        )
                        annotation_list.append(sc_annotation)
                continue

            label_id = label_name_to_id_mapping.get(label_name)
            label = {"id": label_id, "probability": 1.0}
            if (
                self.task_type not in GLOBAL_TASK_TYPES
                or preserve_shape_for_global_labels
            ):
                if isinstance(annotation, Bbox):
                    x1 = float(annotation.points[0])
                    y1 = float(annotation.points[1])
                    x2 = float(annotation.points[2])
                    y2 = float(annotation.points[3])
                    shape = {
                        "type": "RECTANGLE",
                        "x": x1,
                        "y": y1,
                        "width": x2 - x1,
                        "height": y2 - y1,
                    }
                elif isinstance(annotation, Polygon):
                    if self.task_type == TaskType.ROTATED_DETECTION:
                        contour = np.array(
                            [
                                np.array([x, y], dtype=np.float32)
                                for x, y in zip(*[iter(annotation.points)] * 2)
                            ],
                            dtype=np.float32,
                        )
                        min_rect = cv2.minAreaRect(contour)
                        shape = {
                            "type": "ROTATED_RECTANGLE",
                            "x": min_rect[0][0],
                            "y": min_rect[0][1],
                            "width": min_rect[1][0],
                            "height": min_rect[1][1],
                            "angle": min_rect[2],
                        }
                    else:
                        points = [
                            {"x": float(x), "y": float(y)}
                            for x, y in zip(*[iter(annotation.points)] * 2)
                        ]
                        shape = {"type": "POLYGON", "points": points}
                else:
                    logging.warning(
                        f"Unsupported annotation type found: "
                        f"{type(annotation)}. Skipping..."
                    )
                    continue
                sc_annotation = AnnotationRESTConverter.annotation_from_dict(
                    {"labels": [label], "shape": shape}
                )
                annotation_list.append(sc_annotation)
            else:
                labels.append(label)

        if not preserve_shape_for_global_labels and self.task_type in GLOBAL_TASK_TYPES:
            shape = {
                "type": "RECTANGLE",
                "x": 0.0,
                "y": 0.0,
                "width": float(image_size[1]),
                "height": float(image_size[0]),
            }
            sc_annotation = AnnotationRESTConverter.annotation_from_dict(
                {"labels": labels, "shape": shape}
            )
            annotation_list.append(sc_annotation)
        return annotation_list

    @property
    def applied_filters(self) -> List[Dict[str, Union[List[str], str]]]:
        """
        Return a list of filters and their parameters that have been previously
        applied to the dataset.
        """
        return copy.deepcopy(self._applied_filters)

    def filter_dataset(self, labels: Sequence[str], criterion="OR") -> None:
        """
        Retain only those items with annotations in the list of labels passed.

        :param: labels     List of labels to filter on
        :param: criterion  Filter criterion, currently "OR" or "AND" are implemented
        """
        self.dataset.filter_items_by_labels(labels=labels, criterion=criterion)
        self._applied_filters.append({"labels": labels, "criterion": criterion})

    def group_labels(self, labels_to_group: List[str], group_name: str) -> None:
        """
        Group multiple labels into one. Grouping converts the list of labels into one
        single label named `group_name`.

        This method does not return anything, but instead overrides the label map for
        the datamaro dataset to account for the grouping.

        :param labels_to_group: List of labels names that should be grouped together
        :param group_name: Name of the resulting label
        :return:
        """
        label_keys = [
            get_dict_key_from_value(self.datum_label_map, label)
            for label in labels_to_group
        ]
        new_label_map = copy.deepcopy(self.datum_label_map)
        for key in label_keys:
            new_label_map[key] = group_name
        self.override_label_map(new_label_map=new_label_map)

    def get_annotation_stats(self) -> Dict[str, Dict[str, int]]:
        """
        Return the object and image counts per label in the dataset.

        :return: Dictionary containing label names as keys, and as values:
            - n_images: Number of images containing this label
            - n_objects: Number of independent objects with this label
        """
        label_statistics: Dict[str, Dict[str, int]] = {}
        for item in self.dataset.dataset:
            label_names = set(
                [
                    self.datum_label_map[annotation.label]
                    for annotation in item.annotations
                ]
            )
            object_counts: Dict[str, int] = {}
            for label in label_names:
                object_counts[label] = 0
            for annotation in item.annotations:
                label_name = self.datum_label_map[annotation.label]
                object_counts[label_name] += 1

            for label in label_names:
                label_stats = label_statistics.get(label, {})
                if not label_stats:
                    label_statistics[label] = {
                        "n_images": 1,
                        "n_objects": object_counts[label],
                    }
                else:
                    label_statistics[label]["n_images"] += 1
                    label_statistics[label]["n_objects"] += object_counts[label]
        return label_statistics
