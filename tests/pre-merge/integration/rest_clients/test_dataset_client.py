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
import warnings
from typing import List

import pytest

from geti_sdk.rest_clients import DatasetClient
from tests.helpers import ProjectService
from tests.helpers.constants import PROJECT_PREFIX


class TestDatasetClient:
    @pytest.mark.vcr()
    def test_create_and_delete_dataset(
        self, fxt_project_service: ProjectService, fxt_default_labels: List[str]
    ) -> None:
        """
        Verifies that creating a new dataset in an existing project works
        """
        # Arrange
        project = fxt_project_service.create_project(
            project_name=f"{PROJECT_PREFIX}_dataset_client",
            project_type="detection",
            labels=[fxt_default_labels],
        )
        dataset_client = DatasetClient(
            session=fxt_project_service.session,
            workspace_id=fxt_project_service.workspace_id,
            project=project,
        )
        old_datasets = dataset_client.get_all_datasets()
        test_dataset_name = "test_dataset"
        test_dataset_2_name = test_dataset_name + "_2"

        # Act
        test_dataset = dataset_client.create_dataset(name=test_dataset_name)
        test_dataset_2 = dataset_client.create_dataset(name=test_dataset_2_name)

        # Assert
        new_datasets = dataset_client.get_all_datasets()
        assert test_dataset in new_datasets
        assert test_dataset_2 in new_datasets
        for dataset in old_datasets:
            assert dataset in new_datasets
        test_dataset_by_name = dataset_client.get_dataset_by_name(test_dataset_name)
        assert test_dataset == test_dataset_by_name

        # Delete existing dataset
        dataset_client.delete_dataset(dataset=test_dataset)
        # Delete non-existing dataset
        with warnings.catch_warnings():
            dataset_client.delete_dataset(dataset=test_dataset)

        # Assert
        new_datasets = dataset_client.get_all_datasets()
        assert test_dataset in old_datasets
        for dataset in old_datasets:
            if dataset != test_dataset:
                assert dataset in new_datasets
            else:
                assert dataset not in new_datasets
