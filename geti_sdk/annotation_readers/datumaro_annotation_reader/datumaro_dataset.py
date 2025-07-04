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
import logging
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from datumaro.components.annotation import (
    AnnotationType,
    LabelCategories,
    PointsCategories,
)
from datumaro.components.dataset import Dataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.environment import Environment

from geti_sdk.data_models import TaskType
from geti_sdk.utils import deprecate, get_dict_key_from_value


@deprecate(
    since="2.12",
    use="GetiIE",
    reason="Dataset and annotations can be imported using Geti's native Dataset Import/Export functionality.",
)
class DatumaroDataset(object):
    """
    Wrapper for interacting with the datumaro dataset, contains some example
    functions for dataset operations that can be carried out prior to importing the
    dataset into an Intel® Geti™ project.
    """

    def __init__(self, dataset_format: str, dataset_path: str):
        """
        Initialize the datumaro dataset.

        :param dataset_format: string containing the format of the dataset, i.e.
            'voc', 'coco', 'imagenet', etc.
        :param dataset_path: Path to the dataset on local disk
        """
        self.dataset_format = dataset_format
        self.dataset_path = dataset_path
        self.dataset, self.environment = self.create_datumaro_dataset()
        self._subset_names = self.dataset.subsets().keys()
        self._filtered_categories = [self.dataset.categories()[AnnotationType.label]]
        if self.dataset.categories().get(AnnotationType.points):
            self._filtered_categories.append(
                self.dataset.categories()[AnnotationType.points]
            )

    def prepare_dataset(
        self, task_type: TaskType, previous_task_type: Optional[TaskType] = None
    ) -> Dataset:
        """
        Prepare the dataset for uploading to Intel Geti.

        :param task_type: TaskType to prepare the dataset for
        :param previous_task_type: Optional type of the (trainable) task preceding
            the current task in the pipeline. This is only used for global tasks
        """
        if task_type.is_detection and task_type != TaskType.ROTATED_DETECTION:
            new_dataset = self.dataset.transform(
                self.dataset.env.transforms.get("shapes_to_boxes")
            )
            logging.info("Annotations have been converted to boxes")
        elif task_type.is_segmentation or task_type == TaskType.ROTATED_DETECTION:
            converted_dataset = self.dataset.transform(
                self.dataset.env.transforms.get("masks_to_polygons")
            )
            new_dataset = converted_dataset.filter(
                '/item/annotation[type="polygon"]', filter_annotations=True
            )
            logging.info("Annotations have been converted to polygons")
        elif task_type.is_keypoint_detection:
            new_dataset = self.dataset
            logging.info("Keypoint detection dataset prepared.")
        elif task_type.is_global:
            if previous_task_type is not None and (
                previous_task_type.is_segmentation
                or previous_task_type == TaskType.ROTATED_DETECTION
            ):
                converted_dataset = self.dataset.transform(
                    self.dataset.env.transforms.get("masks_to_polygons")
                )
                new_dataset = converted_dataset.filter(
                    '/item/annotation[type="polygon"]', filter_annotations=True
                )
            else:
                new_dataset = self.dataset.transform(
                    self.dataset.env.transforms.get("shapes_to_boxes")
                )
            logging.info(f"{str(task_type).capitalize()} dataset prepared.")
        else:
            raise ValueError(f"Unsupported task type {task_type}")
        return new_dataset

    def set_dataset(self, dataset: Dataset):
        """
        Set the dataset for this DatumaroDataset instance.

        :param dataset:
        :return:
        """
        self.dataset = dataset
        self.environment = dataset.env
        self._subset_names = self.dataset.subsets().keys()

    @property
    def label_categories(self) -> LabelCategories:
        """
        Return the LabelCategories in the dataset.
        """
        return self._filtered_categories[0]

    @property
    def label_names(self) -> List[str]:
        """
        Return a list of all label names in the dataset.
        """
        return [item.name for item in self.label_categories]

    @property
    def label_mapping(self) -> Dict[int, str]:
        """
        Return the mapping of label index to label name.
        """
        if len(self._filtered_categories) == 2:
            # If there are points categories, we need to use them as labels
            return self.points_mapping
        return {value: key for key, value in self.label_categories._indices.items()}

    @property
    def points_categories(self) -> Union[PointsCategories, None]:
        """
        Return the PointCategories in the dataset.
        """
        try:
            return self._filtered_categories[1]
        except IndexError:
            logging.info(
                "No points categories found in the dataset. Please check the dataset "
                "format and ensure that it contains points annotations."
            )
            return None

    @property
    def points_names(self) -> Union[List[str], None]:
        """
        Return a list of all point names in the dataset.
        """
        if self.points_categories:
            return self.points_categories.items[0].labels
        return None

    @property
    def points_mapping(self) -> Union[Dict[int, str], None]:
        """
        Return the mapping of point index to label name.
        """
        if self.points_categories:
            return {
                idx: name
                for idx, name in enumerate(self.points_categories.items[0].labels)
            }
        logging.info(
            "No points categories found in the dataset. Please check the dataset "
        )
        return None

    @property
    def joints_mapping(self) -> set[tuple[int, int]]:
        """
        Return the mapping of point index to label name.
        """
        return self.points_categories.items[0].joints

    @property
    def image_names(self) -> List[str]:
        """
        Return the list of media names included in the dataset.
        """
        return [item.id for item in self.dataset]

    def create_datumaro_dataset(self) -> Tuple[Dataset, Environment]:
        """
        Initialize a datumaro dataset.
        """
        t_start = time.time()
        dataset = Dataset.import_from(
            path=self.dataset_path, format=self.dataset_format
        )
        logging.info(
            f"Datumaro dataset consisting of {len(dataset)} items in "
            f"{self.dataset_format} format was loaded from {self.dataset_path}"
        )
        logging.info(
            f"Datumaro dataset was created in {time.time() - t_start:.1f} seconds"
        )
        return dataset, dataset.env

    def remove_unannotated_items(self):
        """
        Keep only annotated images.
        """
        self.dataset = self.dataset.select(lambda item: len(item.annotations) != 0)

    def filter_items_by_labels(self, labels: Sequence[str], criterion="OR") -> None:
        """
        Retain only those items with annotations in the list of labels passed.

        :param labels: List of labels to filter on
        :param criterion: Filter criterion, currently "OR", "NOT", "AND" and "XOR" are
            implemented
        """
        label_map = self.label_mapping
        # Sanity check for filtering
        for label in labels:
            if label not in list(label_map.values()):
                raise ValueError(
                    f"Cannot filter on label {label} because this is not in the "
                    f"dataset."
                )

        if labels:

            def select_function(dataset_item: DatasetItem, labels: List[str]):
                # Filter function to apply to each item in the dataset
                item_labels = [label_map[x.label] for x in dataset_item.annotations]
                matches = []
                for label in labels:
                    if label in item_labels:
                        if criterion == "OR":
                            return True
                        elif criterion in ["AND", "NOT", "XOR"]:
                            matches.append(True)
                        else:
                            raise ValueError(
                                'Invalid filter criterion, please select "OR", "NOT", '
                                '"XOR", or "AND".'
                            )
                    else:
                        matches.append(False)
                if criterion == "AND":
                    return all(matches)
                elif criterion == "NOT":
                    return not any(matches)
                elif criterion == "XOR":
                    return np.sum(matches) == 1

            # Messy way to manually keep track of labels and indices, must be a
            # better way in Datumaro but haven't found it yet
            label_categories = LabelCategories.from_iterable(labels)
            new_labelmap = {}
            for label in labels:
                label_key = get_dict_key_from_value(label_map, label)
                new_labelmap[label_key] = label
            label_categories._indices = {v: k for k, v in new_labelmap.items()}
            # Filter and create a new dataset to update the dataset categories
            self.dataset = Dataset.from_iterable(
                self.dataset.select(lambda item: select_function(item, labels)),
                categories=self.dataset.categories(),
                env=self.dataset.env,
            )
            logging.info(
                f"After filtering, dataset with labels {labels} contains "
                f"{len(self.dataset)} items."
            )
            self._filtered_categories = [label_categories]
            if self.dataset.categories().get(AnnotationType.points):
                self._filtered_categories.append(
                    self.dataset.categories()[AnnotationType.points]
                )

    def __get_item_by_id_from_subsets(
        self, datum_id: str, search_by_name: bool = False
    ) -> Optional[DatasetItem]:
        """
        Search all subsets for the item with id `datum_id`

        :param datum_id: Datumaro id of the item to retrieve
        :param search_by_name: True to search for the image by filename as well, in
            addition to searching within the datumaro dataset
        :return: Dataset item with the given id, or None if the item was not found
        """
        ds_item: Optional[DatasetItem] = None
        for subset_name in self._subset_names:
            ds_item = self.dataset.get(id=datum_id, subset=subset_name)
        if search_by_name and ds_item is None:
            for subset_name in self._subset_names:
                image_relative_path = self.__search_image_by_filename(
                    image_filename=datum_id, subset_name=subset_name
                )
                if image_relative_path is not None:
                    ds_item = self.dataset.get(
                        id=image_relative_path, subset=subset_name
                    )
        return ds_item

    def __search_image_by_filename(
        self, image_filename: str, subset_name: str
    ) -> Optional[str]:
        """
        Search for an image with name `image_filename` inside the directory tree in
        the data path.

        :param image_filename: Filename of the image to search for (without extension!)
        :param subset_name: Name of the subset which the image is in
        :return: Datumaro id (expressed as a relative unix-style path) for the image,
            defined with respect to the subset it is in. If no matches are found for the
            filename, this method returns None
        """
        matches = glob.glob(
            os.path.join(
                self.dataset_path,
                "**",
                "images",
                subset_name,
                "**",
                f"{image_filename}.*",
            ),
            recursive=True,
        )
        if len(matches) > 1:
            logging.warning(
                f"Multiple images with filename '{image_filename}' found in dataset "
                f"subset '{subset_name}', unable to uniquely identify dataset item"
            )
            return None
        elif len(matches) == 0:
            return None
        relative_path = matches[0].split(f"{subset_name}{os.path.sep}")[-1]
        path = os.path.normpath(relative_path)
        path = os.path.splitext(path)[0]
        path_components = path.split(os.path.sep)
        return "/".join(path_components)

    def get_item_by_id(self, datum_id: str) -> DatasetItem:
        """
        Return the dataset item by its id.

        :param datum_id: Datumaro id of the item to retrieve
        :raises: ValueError if no item by that id was found.
        :return: DatasetItem with the given id
        """
        ds_item = self.__get_item_by_id_from_subsets(
            datum_id=datum_id, search_by_name=True
        )
        if ds_item is None:
            raise ValueError(
                f"Unable to identify dataset item with id {datum_id} in the dataset. "
                f"Please try to simplify the internal filestructure of the dataset, "
                f"and make sure that images names (within subsets) are unique."
            )
        return ds_item
