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


import pytest


@pytest.fixture()
def fxt_hierarchical_classification_labels() -> list[dict[str, str]]:
    yield [
        {"name": "animal", "group": "animal"},
        {"name": "dog", "parent_id": "animal", "group": "species"},
        {"name": "cat", "parent_id": "animal", "group": "species"},
        {"name": "vehicle", "group": "vehicle"},
        {"name": "car", "parent_id": "vehicle", "group": "vehicle type"},
        {"name": "taxi", "parent_id": "vehicle", "group": "vehicle type"},
        {"name": "truck", "parent_id": "vehicle", "group": "vehicle type"},
        {"name": "red", "parent_id": "vehicle", "group": "vehicle color"},
        {"name": "blue", "parent_id": "vehicle", "group": "vehicle color"},
        {"name": "black", "parent_id": "vehicle", "group": "vehicle color"},
        {"name": "grey", "parent_id": "vehicle", "group": "vehicle color"},
    ]


@pytest.fixture()
def fxt_default_labels() -> list[str]:
    yield ["cube", "cylinder"]


@pytest.fixture()
def fxt_default_keypoint_labels() -> list[str]:
    yield [
        "left_shoulder",
        "left_wrist",
        "left_ankle",
        "left_ear",
        "left_elbow",
        "left_knee",
        "left_hip",
        "left_eye",
        "right_shoulder",
        "right_wrist",
        "right_ankle",
        "right_ear",
        "right_elbow",
        "right_knee",
        "right_hip",
        "right_eye",
        "nose",
    ]


@pytest.fixture()
def fxt_light_bulbs_labels() -> list[str]:
    yield ["On", "Off"]
