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
from collections.abc import Callable
from typing import Any

import pytest

from geti_sdk.data_models import (
    Annotation,
    AnnotationScene,
    Image,
    ScoredLabel,
    Video,
    VideoFrame,
)
from geti_sdk.data_models.containers import MediaList
from geti_sdk.data_models.media_identifiers import ImageIdentifier
from geti_sdk.data_models.shapes import Point, Polygon, Rectangle


@pytest.fixture()
def fxt_normalized_annotation_dict() -> dict[str, Any]:
    yield {
        "labels": [{"probability": 1.0, "name": "Dog", "color": "#000000ff"}],
        "shape": {
            "x": 0.05,
            "y": 0.1,
            "width": 0.9,
            "height": 0.8,
            "type": "RECTANGLE",
        },
        "labels_to_revisit": [],
    }


@pytest.fixture()
def fxt_rectangle_annotation_factory(
    fxt_scored_label: ScoredLabel,
) -> Callable[[int, int], Annotation]:
    def _generate_rectangle_annotation(image_width: int = 1000, image_height: int = 2000) -> Annotation:
        return Annotation(
            shape=Rectangle(
                x=0.05 * image_width,
                y=0.1 * image_height,
                width=0.9 * image_width,
                height=0.8 * image_height,
            ),
            labels=[fxt_scored_label],
        )

    return _generate_rectangle_annotation


@pytest.fixture()
def fxt_normalized_annotation_polygon_dict() -> dict[str, Any]:
    yield {
        "labels": [{"probability": 1.0, "name": "Dog", "color": "#000000ff"}],
        "shape": {
            "points": [
                {"x": 0.1, "y": 0.2},
                {"x": 0.1, "y": 0.4},
                {"x": 0.5, "y": 0.4},
                {"x": 0.5, "y": 0.2},
            ],
            "type": "POLYGON",
        },
        "labels_to_revisit": [],
    }


@pytest.fixture()
def fxt_polygon_annotation_factory(
    fxt_scored_label: ScoredLabel,
) -> Callable[[int, int], Annotation]:
    def _generate_polygon_annotation(image_width: int = 1000, image_height: int = 2000) -> Annotation:
        return Annotation(
            shape=Polygon(
                points=[
                    Point(x=int(0.1 * image_width), y=int(0.2 * image_height)),
                    Point(x=int(0.1 * image_width), y=int(0.4 * image_height)),
                    Point(x=int(0.5 * image_width), y=int(0.4 * image_height)),
                    Point(x=int(0.5 * image_width), y=int(0.2 * image_height)),
                ]
            ),
            labels=[fxt_scored_label],
        )

    return _generate_polygon_annotation


@pytest.fixture()
def fxt_annotation_scene(
    fxt_geti_image: Image,
    fxt_rectangle_annotation_factory,
    fxt_polygon_annotation_factory,
) -> AnnotationScene:
    width = fxt_geti_image.media_information.width
    height = fxt_geti_image.media_information.height
    yield AnnotationScene(
        annotations=[
            fxt_rectangle_annotation_factory(width, height),
            fxt_polygon_annotation_factory(width, height),
        ],
        media_identifier=fxt_geti_image.identifier,
    )


@pytest.fixture()
def fxt_video_annotation_scenes(
    fxt_geti_video: Video,
    fxt_rectangle_annotation_factory,
    fxt_polygon_annotation_factory,
    fxt_video_frames: MediaList[VideoFrame],
) -> list[AnnotationScene]:
    width = fxt_geti_video.media_information.width
    height = fxt_geti_video.media_information.height
    yield [
        AnnotationScene(
            annotations=[
                fxt_rectangle_annotation_factory(width, height),
                fxt_polygon_annotation_factory(width, height),
            ],
            media_identifier=video_frame.identifier,
        )
        for video_frame in fxt_video_frames
    ]


@pytest.fixture()
def fxt_annotation_scene_from_normalized(
    fxt_rectangle_annotation_factory,
    fxt_polygon_annotation_factory,
    fxt_image_identifier: ImageIdentifier,
) -> AnnotationScene:
    img_width = 1000
    img_height = 2000
    yield AnnotationScene(
        annotations=[
            fxt_rectangle_annotation_factory(img_width, img_height),
            fxt_polygon_annotation_factory(img_width, img_height),
        ],
        media_identifier=fxt_image_identifier,
    )
