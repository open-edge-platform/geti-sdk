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
import os
import warnings
from collections.abc import Sequence
from glob import glob

from geti_sdk.annotation_readers import AnnotationReader
from geti_sdk.data_models import Annotation, ScoredLabel, TaskType
from geti_sdk.data_models.enums.media_type import SUPPORTED_IMAGE_FORMATS
from geti_sdk.data_models.media import MediaInformation
from geti_sdk.data_models.shapes import Rectangle


class DirectoryTreeAnnotationReader(AnnotationReader):
    """
    AnnotationReader for loading single label classification annotations from a
    dataset organized in a directory tree. This annotation reader expects images to
    be put in folders, where the name of each image folder corresponds to the label
    that should be assigned to all images inside it.

    :param base_data_folder: Root of the directory tree that contains the dataset
    :param subset_folder_names: Optional list of subfolders of the base_data_folder
        that should not be considered as labels, but should be used to acquire the
        data. For example ['train', 'validation', 'test'] for a dataset that is split
        into three subsets.
    :param task_type: TaskType for the task in the Intel® Geti™ platform to which the
        annotations should be uploaded
    """

    def __init__(
        self,
        base_data_folder: str,
        subset_folder_names: Sequence[str] | None = None,
        task_type: TaskType | str = TaskType.CLASSIFICATION,
    ):
        if not task_type.is_global:
            raise ValueError("The DirectoryTreeAnnotationReader only supports annotations for global task types.")
        super().__init__(
            base_data_folder=base_data_folder,
            annotation_format="directory",
            task_type=task_type,
        )

        has_root_level_subsets = subset_folder_names is not None
        if has_root_level_subsets:
            self.target_data_dirs = [os.path.join(base_data_folder, subset) for subset in subset_folder_names]
        else:
            self.target_data_dirs = [base_data_folder]
        # Label map is a dictionary mapping the root labels to new label names. It
        # can be used to filter or group the labels in the dataset
        self.has_filters_or_grouping = False
        self._original_labels = self.get_all_label_names()
        self._label_map: dict[str, str] = {label: label for label in self._original_labels}

    @property
    def label_map(self) -> dict[str, str]:
        """
        Return the label map for the dataset, mapping the root label names (keys) to
        potential new label names (values). It is used to filter or group the dataset.

        If no filters or grouping has been applied, it returns a dictionary with key,
        value pairs that have identical keys and values, i.e. {"dog": "dog"}
        """
        return self._label_map

    def reset_filters_and_grouping(self):
        """
        Reset the applied filters and grouping, to recover the original dataset
        """
        self._label_map = {label: label for label in self._original_labels}
        self.has_filters_or_grouping = False

    def get_data(
        self,
        filename: str,
        label_name_to_id_mapping: dict,
        media_information: MediaInformation,
        preserve_shape_for_global_labels: bool = False,
        image_name_as_full_path: bool = False,
    ) -> list[Annotation]:
        """
        Return the list of annotations for the media item with name `filename`

        :param filename: Name of the item to return the annotations for
        :param label_name_to_id_mapping: Dictionary mapping the name of a label to its
            unique database ID
        :param media_information: MediaInformation object containing information
            (e.g. width, height) about the media item to upload the annotation for
        :param preserve_shape_for_global_labels: Unused parameter in this type of
            annotation reader
        :param image_name_as_full_path: Set to True if the `filename` contains the
            full path to the image
        :return: A list of Annotation objects for the media item
        """
        filepath = ""
        annotations: list[Annotation] = []
        if image_name_as_full_path:
            label_matches = [os.path.basename(os.path.dirname(filename))]
            extension = filename[:-4]
            if extension in SUPPORTED_IMAGE_FORMATS:
                filepath = filename
            else:
                for format_extension in SUPPORTED_IMAGE_FORMATS:
                    full_name = filename + format_extension
                    if os.path.isfile(full_name):
                        filepath = full_name
                if filepath == "":
                    raise ValueError(
                        f"No valid image file found at path {filename}, unable to generate annotation data."
                    )
        else:
            matches = glob(os.path.join(self.base_folder, "**", f"{filename}.*"), recursive=True)
            label_matches = [os.path.basename(os.path.dirname(match)) for match in matches]
            if len(label_matches) > 1:
                warnings.warn(
                    f"Multiple matching labels found for image with "
                    f"name {filename}: {label_matches}. Skipping this image..."
                )
            elif len(label_matches) == 0:
                logging.info(
                    f"Image with name {filename} was not found in the dataset at path "
                    f"{self.base_folder}. Skipping this image..."
                )
                return []
            filepath = matches[0]
        label_name = self.label_map[label_matches[0]]
        label = ScoredLabel(
            name=label_name,
            probability=1.0,
            id=label_name_to_id_mapping[label_name],
        )
        annotations.append(
            Annotation(
                labels=[label],
                shape=Rectangle(
                    x=0,
                    y=0,
                    width=media_information.width,
                    height=media_information.height,
                ),
            )
        )
        return annotations

    def get_all_label_names(self) -> list[str]:
        """
        Identify all label names contained in the dataset
        """
        label_names: set[str] = set()
        for directory in self.target_data_dirs:
            for path, sub_directories, files in os.walk(directory):
                for sub_directory in sub_directories:
                    if self.has_filters_or_grouping and sub_directory not in self.label_map:
                        continue
                    label_name = self.label_map[sub_directory] if self.has_filters_or_grouping else sub_directory
                    label_names.add(label_name)
        return list(label_names)

    def get_data_filenames(self) -> list[str]:
        """
        Return a list of annotated media files found in the dataset.

        :return: List of filenames (excluding extension) for all annotated files in
            the data folder
        """
        data_file_paths: list[str] = []
        for directory in self.target_data_dirs:
            for path, sub_directories, files in os.walk(directory):
                for name in files:
                    if self.has_filters_or_grouping and os.path.basename(path) not in self.label_map:
                        continue
                    data_file_paths.append(os.path.join(path, os.path.splitext(name)[0]))
        return data_file_paths

    def filter_dataset(self, labels: Sequence[str], criterion: str = "OR") -> None:
        """
        Retain only those items with annotations in the list of labels passed.

        :param labels: List of labels to filter on
        :param criterion: Unused parameter for this type of annotation reader
        """
        self._label_map = {label: label for label in labels}
        self.applied_filters.append({"labels": labels, "criterion": criterion})
        self.has_filters_or_grouping = True

    def group_labels(self, labels_to_group: list[str], group_name: str) -> None:
        """
        Group multiple labels into one. Grouping converts the list of labels into one
        single label named `group_name`.

        This method does not return anything, but instead overrides the label map for
        the annotation reader to account for the grouping.

        :param labels_to_group: List of labels names that should be grouped together
        :param group_name: Name of the resulting label
        """
        for org_label_name, mapped_label in self.label_map.items():
            if org_label_name in labels_to_group:
                self._label_map.update({org_label_name: group_name})
        self.has_filters_or_grouping = True

    def get_annotation_stats(self) -> dict[str, dict[str, int]]:
        """
        Return the image counts per label in the dataset.

        :return: Dictionary containing label names as keys, and as values:
            - n_images: Number of images containing this label
        """
        label_statistics: dict[str, dict[str, int]] = {}
        label_names = self.get_all_label_names()
        for label in label_names:
            label_statistics[label] = {"n_images": 0}
        for item_filepath in self.get_data_filenames():
            item_label = self.label_map[os.path.basename(os.path.dirname(item_filepath))]
            label_statistics[item_label]["n_images"] += 1
        return label_statistics
