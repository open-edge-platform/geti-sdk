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

from typing import Any

import pytest

from geti_sdk.data_models import Project
from geti_sdk.rest_converters import ProjectRESTConverter


@pytest.fixture()
def fxt_nightly_projects_rest() -> list[dict[str, Any]]:
    yield [
        {
            "name": "geti_sdk_test_nightly_classification",
            "pipeline": {
                "tasks": [
                    {
                        "title": "Dataset",
                        "task_type": "dataset",
                        "id": "636b963c308ea65372b802b9",
                    },
                    {
                        "title": "Classification task",
                        "task_type": "classification",
                        "labels": [
                            {
                                "name": "cube",
                                "color": "#97bcb2ff",
                                "group": "classification task label group",
                                "is_empty": False,
                                "hotkey": "",
                                "id": "636b963c308ea65372b802c2",
                                "parent_id": None,
                                "is_anomalous": False,
                            },
                            {
                                "name": "cylinder",
                                "color": "#ccc677ff",
                                "group": "classification task label group",
                                "is_empty": False,
                                "hotkey": "",
                                "id": "636b963c308ea65372b802c4",
                                "parent_id": None,
                                "is_anomalous": False,
                            },
                        ],
                        "label_schema_id": "636b963c308ea65372b802c5",
                        "id": "636b963c308ea65372b802ba",
                    },
                ],
                "connections": [
                    {
                        "to": "636b963c308ea65372b802ba",
                        "from": "636b963c308ea65372b802b9",
                    }
                ],
            },
            "datasets": [
                {
                    "name": "Dataset",
                    "id": "636b963c308ea65372b802bc",
                    "creation_time": "2022-11-09T11:59:56.034000+00:00",
                    "use_for_training": True,
                }
            ],
            "score": None,
            "performance": None,
            "creation_time": "2022-11-09T11:59:56.036000+00:00",
            "id": "636b963c308ea65372b802bb",
            "thumbnail": "/api/v1/workspaces/633dad214155584fcd26b41a/projects/636b963c308ea65372b802bb/thumbnail",
            "creator_id": "admin@sc-project.intel.com",
        },
        {
            "name": "geti_sdk_test_nightly_detection",
            "pipeline": {
                "tasks": [
                    {
                        "title": "Dataset",
                        "task_type": "dataset",
                        "id": "636b9717308ea65372b80320",
                    },
                    {
                        "title": "Detection task",
                        "task_type": "detection",
                        "labels": [
                            {
                                "name": "cube",
                                "color": "#fb09a4ff",
                                "group": "detection task label group",
                                "is_empty": False,
                                "hotkey": "",
                                "id": "636b9717308ea65372b80329",
                                "parent_id": None,
                                "is_anomalous": False,
                            },
                            {
                                "name": "cylinder",
                                "color": "#58523dff",
                                "group": "detection task label group",
                                "is_empty": False,
                                "hotkey": "",
                                "id": "636b9717308ea65372b8032b",
                                "parent_id": None,
                                "is_anomalous": False,
                            },
                            {
                                "name": "No Object",
                                "color": "#b36309ff",
                                "group": "No Object",
                                "is_empty": True,
                                "hotkey": "",
                                "id": "636b9717308ea65372b8032c",
                                "parent_id": None,
                                "is_anomalous": False,
                            },
                        ],
                        "label_schema_id": "636b9717308ea65372b8032e",
                        "id": "636b9717308ea65372b80321",
                    },
                ],
                "connections": [
                    {
                        "to": "636b9717308ea65372b80321",
                        "from": "636b9717308ea65372b80320",
                    }
                ],
            },
            "datasets": [
                {
                    "name": "Dataset",
                    "id": "636b9717308ea65372b80323",
                    "creation_time": "2022-11-09T12:03:35.075000+00:00",
                    "use_for_training": True,
                }
            ],
            "score": None,
            "performance": None,
            "creation_time": "2022-11-09T12:03:35.076000+00:00",
            "id": "636b9717308ea65372b80322",
            "thumbnail": "/api/v1/workspaces/633dad214155584fcd26b41a/projects/636b9717308ea65372b80322/thumbnail",
            "creator_id": "admin@sc-project.intel.com",
        },
        {
            "name": "geti_sdk_test_nightly_detection_to_classification",
            "pipeline": {
                "tasks": [
                    {
                        "title": "Dataset",
                        "task_type": "dataset",
                        "id": "636b9847308ea65372b803b2",
                    },
                    {
                        "title": "Detection task",
                        "task_type": "detection",
                        "labels": [
                            {
                                "name": "block",
                                "color": "#700bd0ff",
                                "group": "detection task label group",
                                "is_empty": False,
                                "hotkey": "",
                                "id": "636b9847308ea65372b803c1",
                                "parent_id": None,
                                "is_anomalous": False,
                            },
                            {
                                "name": "No Object",
                                "color": "#30f36cff",
                                "group": "No Object",
                                "is_empty": True,
                                "hotkey": "",
                                "id": "636b9847308ea65372b803c3",
                                "parent_id": None,
                                "is_anomalous": False,
                            },
                        ],
                        "label_schema_id": "636b9847308ea65372b803c5",
                        "id": "636b9847308ea65372b803b3",
                    },
                    {
                        "title": "Crop task",
                        "task_type": "crop",
                        "labels": None,
                        "label_schema_id": None,
                        "id": "636b9847308ea65372b803b4",
                    },
                    {
                        "title": "Classification task",
                        "task_type": "classification",
                        "labels": [
                            {
                                "name": "cube",
                                "color": "#2d7ddcff",
                                "group": "classification task label group",
                                "is_empty": False,
                                "hotkey": "",
                                "id": "636b9847308ea65372b803c6",
                                "parent_id": "636b9847308ea65372b803c1",
                                "is_anomalous": False,
                            },
                            {
                                "name": "cylinder",
                                "color": "#98c00eff",
                                "group": "classification task label group",
                                "is_empty": False,
                                "hotkey": "",
                                "id": "636b9847308ea65372b803c8",
                                "parent_id": "636b9847308ea65372b803c1",
                                "is_anomalous": False,
                            },
                        ],
                        "label_schema_id": "636b9847308ea65372b803c9",
                        "id": "636b9847308ea65372b803b5",
                    },
                ],
                "connections": [
                    {
                        "to": "636b9847308ea65372b803b3",
                        "from": "636b9847308ea65372b803b2",
                    },
                    {
                        "to": "636b9847308ea65372b803b4",
                        "from": "636b9847308ea65372b803b3",
                    },
                    {
                        "to": "636b9847308ea65372b803b5",
                        "from": "636b9847308ea65372b803b4",
                    },
                ],
            },
            "datasets": [
                {
                    "name": "Dataset",
                    "id": "636b9847308ea65372b803b7",
                    "creation_time": "2022-11-09T12:08:39.732000+00:00",
                    "use_for_training": True,
                }
            ],
            "score": None,
            "performance": None,
            "creation_time": "2022-11-09T12:08:39.734000+00:00",
            "id": "636b9847308ea65372b803b6",
            "thumbnail": "/api/v1/workspaces/633dad214155584fcd26b41a/projects/636b9847308ea65372b803b6/thumbnail",
            "creator_id": "admin@sc-project.intel.com",
        },
        {
            "name": "geti_sdk_test_nightly_detection_to_segmentation",
            "pipeline": {
                "tasks": [
                    {
                        "title": "Dataset",
                        "task_type": "dataset",
                        "id": "636b9a1b308ea65372b80470",
                    },
                    {
                        "title": "Detection task",
                        "task_type": "detection",
                        "labels": [
                            {
                                "name": "block",
                                "color": "#832d42ff",
                                "group": "detection task label group",
                                "is_empty": False,
                                "hotkey": "",
                                "id": "636b9a1b308ea65372b8047f",
                                "parent_id": None,
                                "is_anomalous": False,
                            },
                            {
                                "name": "No Object",
                                "color": "#608ec2ff",
                                "group": "No Object",
                                "is_empty": True,
                                "hotkey": "",
                                "id": "636b9a1b308ea65372b80481",
                                "parent_id": None,
                                "is_anomalous": False,
                            },
                        ],
                        "label_schema_id": "636b9a1b308ea65372b80483",
                        "id": "636b9a1b308ea65372b80471",
                    },
                    {
                        "title": "Crop task",
                        "task_type": "crop",
                        "labels": None,
                        "label_schema_id": None,
                        "id": "636b9a1b308ea65372b80472",
                    },
                    {
                        "title": "Segmentation task",
                        "task_type": "segmentation",
                        "labels": [
                            {
                                "name": "cube",
                                "color": "#f9d17eff",
                                "group": "segmentation task label group",
                                "is_empty": False,
                                "hotkey": "",
                                "id": "636b9a1b308ea65372b80484",
                                "parent_id": None,
                                "is_anomalous": False,
                            },
                            {
                                "name": "cylinder",
                                "color": "#d8469bff",
                                "group": "segmentation task label group",
                                "is_empty": False,
                                "hotkey": "",
                                "id": "636b9a1b308ea65372b80486",
                                "parent_id": None,
                                "is_anomalous": False,
                            },
                            {
                                "name": "Empty",
                                "color": "#b688f9ff",
                                "group": "Empty",
                                "is_empty": True,
                                "hotkey": "",
                                "id": "636b9a1b308ea65372b80487",
                                "parent_id": None,
                                "is_anomalous": False,
                            },
                        ],
                        "label_schema_id": "636b9a1b308ea65372b80489",
                        "id": "636b9a1b308ea65372b80473",
                    },
                ],
                "connections": [
                    {
                        "to": "636b9a1b308ea65372b80471",
                        "from": "636b9a1b308ea65372b80470",
                    },
                    {
                        "to": "636b9a1b308ea65372b80472",
                        "from": "636b9a1b308ea65372b80471",
                    },
                    {
                        "to": "636b9a1b308ea65372b80473",
                        "from": "636b9a1b308ea65372b80472",
                    },
                ],
            },
            "datasets": [
                {
                    "name": "Dataset",
                    "id": "636b9a1b308ea65372b80475",
                    "creation_time": "2022-11-09T12:16:27.588000+00:00",
                    "use_for_training": True,
                }
            ],
            "score": None,
            "performance": None,
            "creation_time": "2022-11-09T12:16:27.590000+00:00",
            "id": "636b9a1b308ea65372b80474",
            "thumbnail": "/api/v1/workspaces/633dad214155584fcd26b41a/projects/636b9a1b308ea65372b80474/thumbnail",
            "creator_id": "admin@sc-project.intel.com",
        },
        {
            "name": "geti_sdk_test_nightly_instance_segmentation",
            "pipeline": {
                "tasks": [
                    {
                        "title": "Dataset",
                        "task_type": "dataset",
                        "id": "636b9d2b308ea65372b80557",
                    },
                    {
                        "title": "Instance segmentation task",
                        "task_type": "instance_segmentation",
                        "labels": [
                            {
                                "name": "cube",
                                "color": "#e53e16ff",
                                "group": "instance segmentation task label group",
                                "is_empty": False,
                                "hotkey": "",
                                "id": "636b9d2b308ea65372b80560",
                                "parent_id": None,
                                "is_anomalous": False,
                            },
                            {
                                "name": "cylinder",
                                "color": "#f76b71ff",
                                "group": "instance segmentation task label group",
                                "is_empty": False,
                                "hotkey": "",
                                "id": "636b9d2b308ea65372b80562",
                                "parent_id": None,
                                "is_anomalous": False,
                            },
                            {
                                "name": "Empty",
                                "color": "#1897bcff",
                                "group": "Empty",
                                "is_empty": True,
                                "hotkey": "",
                                "id": "636b9d2b308ea65372b80563",
                                "parent_id": None,
                                "is_anomalous": False,
                            },
                        ],
                        "label_schema_id": "636b9d2b308ea65372b80565",
                        "id": "636b9d2b308ea65372b80558",
                    },
                ],
                "connections": [
                    {
                        "to": "636b9d2b308ea65372b80558",
                        "from": "636b9d2b308ea65372b80557",
                    }
                ],
            },
            "datasets": [
                {
                    "name": "Dataset",
                    "id": "636b9d2b308ea65372b8055a",
                    "creation_time": "2022-11-09T12:29:31.106000+00:00",
                    "use_for_training": True,
                }
            ],
            "score": None,
            "performance": None,
            "creation_time": "2022-11-09T12:29:31.107000+00:00",
            "id": "636b9d2b308ea65372b80559",
            "thumbnail": "/api/v1/workspaces/633dad214155584fcd26b41a/projects/636b9d2b308ea65372b80559/thumbnail",
            "creator_id": "admin@sc-project.intel.com",
        },
        {
            "name": "geti_sdk_test_nightly_rotated_detection",
            "pipeline": {
                "tasks": [
                    {
                        "title": "Dataset",
                        "task_type": "dataset",
                        "id": "636b9fbf308ea65372b805e9",
                    },
                    {
                        "title": "Rotated detection task",
                        "task_type": "rotated_detection",
                        "labels": [
                            {
                                "name": "cube",
                                "color": "#1e61b3ff",
                                "group": "rotated detection task label group",
                                "is_empty": False,
                                "hotkey": "",
                                "id": "636b9fbf308ea65372b805f2",
                                "parent_id": None,
                                "is_anomalous": False,
                            },
                            {
                                "name": "cylinder",
                                "color": "#1ab21fff",
                                "group": "rotated detection task label group",
                                "is_empty": False,
                                "hotkey": "",
                                "id": "636b9fbf308ea65372b805f4",
                                "parent_id": None,
                                "is_anomalous": False,
                            },
                            {
                                "name": "No Object",
                                "color": "#d6f0b8ff",
                                "group": "No Object",
                                "is_empty": True,
                                "hotkey": "",
                                "id": "636b9fbf308ea65372b805f5",
                                "parent_id": None,
                                "is_anomalous": False,
                            },
                        ],
                        "label_schema_id": "636b9fbf308ea65372b805f7",
                        "id": "636b9fbf308ea65372b805ea",
                    },
                ],
                "connections": [
                    {
                        "to": "636b9fbf308ea65372b805ea",
                        "from": "636b9fbf308ea65372b805e9",
                    }
                ],
            },
            "datasets": [
                {
                    "name": "Dataset",
                    "id": "636b9fbf308ea65372b805ec",
                    "creation_time": "2022-11-09T12:40:31.001000+00:00",
                    "use_for_training": True,
                }
            ],
            "score": None,
            "performance": None,
            "creation_time": "2022-11-09T12:40:31.002000+00:00",
            "id": "636b9fbf308ea65372b805eb",
            "thumbnail": "/api/v1/workspaces/633dad214155584fcd26b41a/projects/636b9fbf308ea65372b805eb/thumbnail",
            "creator_id": "admin@sc-project.intel.com",
        },
        {
            "name": "geti_sdk_test_nightly_segmentation",
            "pipeline": {
                "tasks": [
                    {
                        "title": "Dataset",
                        "task_type": "dataset",
                        "id": "636ba25b308ea65372b8067b",
                    },
                    {
                        "title": "Segmentation task",
                        "task_type": "segmentation",
                        "labels": [
                            {
                                "name": "cube",
                                "color": "#1ef86eff",
                                "group": "segmentation task label group",
                                "is_empty": False,
                                "hotkey": "",
                                "id": "636ba25b308ea65372b80684",
                                "parent_id": None,
                                "is_anomalous": False,
                            },
                            {
                                "name": "cylinder",
                                "color": "#34a0daff",
                                "group": "segmentation task label group",
                                "is_empty": False,
                                "hotkey": "",
                                "id": "636ba25b308ea65372b80686",
                                "parent_id": None,
                                "is_anomalous": False,
                            },
                            {
                                "name": "Empty",
                                "color": "#c09845ff",
                                "group": "Empty",
                                "is_empty": True,
                                "hotkey": "",
                                "id": "636ba25b308ea65372b80687",
                                "parent_id": None,
                                "is_anomalous": False,
                            },
                        ],
                        "label_schema_id": "636ba25b308ea65372b80689",
                        "id": "636ba25b308ea65372b8067c",
                    },
                ],
                "connections": [
                    {
                        "to": "636ba25b308ea65372b8067c",
                        "from": "636ba25b308ea65372b8067b",
                    }
                ],
            },
            "datasets": [
                {
                    "name": "Dataset",
                    "id": "636ba25b308ea65372b8067e",
                    "creation_time": "2022-11-09T12:51:39.224000+00:00",
                    "use_for_training": True,
                }
            ],
            "score": None,
            "performance": None,
            "creation_time": "2022-11-09T12:51:39.226000+00:00",
            "id": "636ba25b308ea65372b8067d",
            "thumbnail": "/api/v1/workspaces/633dad214155584fcd26b41a/projects/636ba25b308ea65372b8067d/thumbnail",
            "creator_id": "admin@sc-project.intel.com",
        },
    ]


@pytest.fixture()
def fxt_nightly_projects(fxt_nightly_projects_rest) -> list[Project]:
    yield [ProjectRESTConverter.from_dict(project_rest) for project_rest in fxt_nightly_projects_rest]


@pytest.fixture()
def fxt_classification_project(fxt_nightly_projects: list[Project]) -> Project:
    yield fxt_nightly_projects[0]


@pytest.fixture()
def fxt_detection_to_classification_project(
    fxt_nightly_projects: list[Project],
) -> Project:
    yield fxt_nightly_projects[2]
