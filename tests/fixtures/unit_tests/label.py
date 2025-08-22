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

from datetime import datetime, timezone

import pytest

from geti_sdk.data_models import Label, ScoredLabel
from geti_sdk.data_models.enums.annotation_kind import AnnotationKind
from geti_sdk.data_models.enums.domain import Domain


@pytest.fixture()
def fxt_label() -> Label:
    yield Label(
        name="Dog",
        color="#000000ff",
        group="Default classification group",
        is_empty=False,
    )


@pytest.fixture()
def fxt_scored_label(fxt_label: Label) -> ScoredLabel:
    yield ScoredLabel.from_label(label=fxt_label, probability=1.0)


class DummyValues:
    MEDIA_HEIGHT = 480
    MEDIA_WIDTH = 640
    LABEL_NAMES = ["rectangle", "ellipse", "triangle"]
    CREATOR_NAME = "SC SDK Fixtures"
    CREATION_DATE = datetime.strptime("01-01-1971_00:00:01", "%d-%m-%Y_%H:%M:%S").astimezone(timezone.utc)
    ANNOTATION_SCENE_KIND = AnnotationKind.ANNOTATION
    ANNOTATION_EDITOR_NAME = "editor"
    MODIFICATION_DATE = datetime(2021, 7, 15, tzinfo=timezone.utc)
    X = 0.375
    Y = 0.25
    WIDTH = 0.25
    HEIGHT = 0.125
    UUID = "92497df6-f45a-11eb-9a03-0242ac130003"
    LABEL_PROBABILITY = 1.0
    LABEL_NAME = "dog"
    DETECTION_DOMAIN = Domain.DETECTION
    ANOMALY_DETECTION_DOMAIN = Domain.ANOMALY_DETECTION
    LABEL_HOTKEY = "ctrl+V"
    FRAME_INDEX = 0


@pytest.fixture
def fxt_empty_classification_label():
    yield Label(
        name="Empty classification label",
        id="0",
        domain=Domain.CLASSIFICATION,
        group="Empty label group",
        color="#ff0000",
        is_empty=True,
    )


@pytest.fixture
def fxt_classification_labels(fxt_empty_classification_label):
    yield [
        Label(
            name=name,
            domain=Domain.CLASSIFICATION,
            hotkey=f"CTRL+{index}",
            group="Default classification group",
            color="#ffdddd",
            id=str(index + 1),
            is_empty=False,
        )
        for index, name in enumerate(DummyValues.LABEL_NAMES)
    ] + [fxt_empty_classification_label]


@pytest.fixture
def fxt_empty_detection_label():
    yield Label(
        name="Empty detection label",
        domain=Domain.DETECTION,
        hotkey=DummyValues.LABEL_HOTKEY,
        color="#ff0000",
        id="0",
        group="Empty label group",
        is_empty=True,
    )


@pytest.fixture
def fxt_detection_labels(fxt_empty_detection_label):
    yield [
        Label(
            name=name,
            color="#ff4400",
            group="Default detection group",
            is_empty=False,
            domain=Domain.DETECTION,
            id=str(index + 1),
            hotkey=f"CTRL+{index}",
        )
        for index, name in enumerate(DummyValues.LABEL_NAMES)
    ] + [fxt_empty_detection_label]


@pytest.fixture
def fxt_segmentation_labels(fxt_empty_segmentation_label):
    yield [
        Label(
            name=name,
            color="#ff4400",
            group="Default segmentation group",
            is_empty=False,
            domain=Domain.SEGMENTATION,
            id=str(index + 1),
            hotkey=f"CTRL+{index}",
        )
        for index, name in enumerate(DummyValues.LABEL_NAMES)
    ] + [fxt_empty_segmentation_label]


@pytest.fixture
def fxt_empty_segmentation_label():
    yield Label(
        name="Empty segmentation label",
        domain=Domain.SEGMENTATION,
        color="#ff0000",
        id="0",
        hotkey=DummyValues.LABEL_HOTKEY,
        group="Empty label group",
        is_empty=True,
    )


@pytest.fixture
def fxt_rotated_detection_labels():
    yield [
        Label(
            name=name,
            color="#ff4400",
            group="Default rotated detection group",
            is_empty=False,
            domain=Domain.ROTATED_DETECTION,
            hotkey=f"CTRL+{index}",
            id=str(index + 1),
        )
        for index, name in enumerate(DummyValues.LABEL_NAMES)
    ]


@pytest.fixture
def fxt_empty_rotated_detection_label():
    yield Label(
        name="Empty rotated detection label",
        domain=Domain.ROTATED_DETECTION,
        hotkey=DummyValues.LABEL_HOTKEY,
        id="0",
        group="Empty label group",
        color="#ff0000",
        is_empty=True,
    )


@pytest.fixture
def fxt_anomalous_label():
    yield Label(
        name=DummyValues.LABEL_NAME,
        domain=DummyValues.ANOMALY_DETECTION_DOMAIN,
        color="#ff0000",
        hotkey=DummyValues.LABEL_HOTKEY,
        group="anomal group",
        is_empty=False,
        is_anomalous=True,
    )


@pytest.fixture
def fxt_anomaly_labels_factory():
    def _build_anom_labels(domain: Domain) -> list[Label]:
        if domain not in (
            Domain.ANOMALY_CLASSIFICATION,
            Domain.ANOMALY_SEGMENTATION,
            Domain.ANOMALY_DETECTION,
        ):
            raise ValueError("This fixtures only generates anomaly labels.")
        normal_label = Label(
            name="dummy_normal_label",
            is_empty=False,
            domain=domain,
            color="#00BF00",
            hotkey=DummyValues.LABEL_HOTKEY,
            id="0",
            is_anomalous=False,
            group="normal group",
        )
        anomalous_label = Label(
            name="dummy_anomalous_label",
            is_empty=False,
            domain=domain,
            color="#ff0000",
            id="1",
            hotkey=DummyValues.LABEL_HOTKEY,
            group="anomal group",
            is_anomalous=True,
        )
        return [normal_label, anomalous_label]

    yield _build_anom_labels


@pytest.fixture
def fxt_keypoint_labels():
    yield [
        Label(
            name=name,
            is_empty=False,
            domain=Domain.KEYPOINT_DETECTION,
            color="#00BF00",
            hotkey=DummyValues.LABEL_HOTKEY,
            id=str(index),
            group="keypoint group",
        )
        for index, name in enumerate(DummyValues.LABEL_NAMES)
    ]
