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

import pytest

from geti_sdk.data_models import Project, TaskType
from geti_sdk.utils import (
    deserialize_dictionary,
    generate_classification_labels,
    generate_segmentation_labels,
)
from geti_sdk.utils.serialization_helpers import DataModelMismatchException


class TestUtils:
    def test_deserialize_dictionary(self, fxt_project_dictionary: dict):
        """
        Verifies that deserializing a dictionary to a python object works.

        The test checks that a DataModelMismatchException is raised in case of a missing key.
        It also verifies that the presence of additional keys in the input dictionary is not a problem.
        """

        # Arrange
        object_type = Project

        dictionary_with_extra_key = copy.deepcopy(fxt_project_dictionary)
        dictionary_with_extra_key.update({"invalid_key": "invalidness"})

        dictionary_with_nested_extra_key = copy.deepcopy(fxt_project_dictionary)
        dictionary_with_nested_extra_key["pipeline"].update({"invalid_key": "invalidness"})
        dictionary_with_nested_extra_key["pipeline"]["tasks"][0].update({"another_invalid_key": "invalidness"})

        dictionary_with_missing_key = copy.deepcopy(fxt_project_dictionary)
        dictionary_with_missing_key.pop("pipeline")

        # Act
        project = deserialize_dictionary(input_dictionary=fxt_project_dictionary, output_type=object_type)

        # Assert
        assert project.name == fxt_project_dictionary["name"]
        assert project.get_trainable_tasks()[0].type == TaskType.DETECTION

        # Act and assert
        deserialize_dictionary(input_dictionary=dictionary_with_extra_key, output_type=object_type)

        # Act and assert
        deserialize_dictionary(input_dictionary=dictionary_with_nested_extra_key, output_type=object_type)

        # Act and assert
        with pytest.raises(DataModelMismatchException):
            deserialize_dictionary(input_dictionary=dictionary_with_missing_key, output_type=object_type)

    def test_generate_segmentation_labels(self):
        # Arrange
        label_names = ["dog", "cat"]

        # Act
        new_label_names = generate_segmentation_labels(label_names)

        # Assert
        assert len(new_label_names) == len(label_names)
        for label in label_names:
            assert label not in new_label_names

    def test_generate_classification_labels(self):
        # Arrange
        label_names = ["dog", "cat"]

        # Act
        new_label_names = generate_classification_labels(label_names)
        new_label_names_multilabel = generate_classification_labels(label_names, multilabel=True)

        # Assert
        assert len(new_label_names) == len(label_names)
        assert len(new_label_names_multilabel) == len(label_names)
        label_groups = [label["group"] for label in new_label_names]
        label_groups_multilabel = [label["group"] for label in new_label_names_multilabel]

        assert len(set(label_groups)) == 1
        assert len(set(label_groups_multilabel)) == len(label_names)
