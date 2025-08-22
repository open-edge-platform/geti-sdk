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

from geti_sdk.data_models import AnnotationKind, AnnotationScene
from geti_sdk.data_models.enums import ShapeType
from geti_sdk.rest_converters.annotation_rest_converter import (
    NormalizedAnnotationRESTConverter,
)


class TestNormalizedAnnotationRESTConverter:
    def test_normalized_annotation_from_dict(self, fxt_normalized_annotation_dict):
        # Arrange
        img_width = 1000
        img_heigth = 2000
        norm_x = fxt_normalized_annotation_dict["shape"]["x"]
        norm_y = fxt_normalized_annotation_dict["shape"]["y"]
        norm_width = fxt_normalized_annotation_dict["shape"]["width"]
        norm_height = fxt_normalized_annotation_dict["shape"]["height"]

        # Act
        annotation = NormalizedAnnotationRESTConverter.normalized_annotation_from_dict(
            fxt_normalized_annotation_dict,
            image_width=img_width,
            image_height=img_heigth,
        )

        # Assert
        assert annotation.shape.type == ShapeType.RECTANGLE
        assert annotation.shape.x == int(norm_x * img_width)
        assert annotation.shape.y == int(norm_y * img_heigth)
        assert annotation.shape.width == int(norm_width * img_width)
        assert annotation.shape.height == int(norm_height * img_heigth)

    def test_normalized_polygon_from_dict(self, fxt_normalized_annotation_polygon_dict):
        # Arrange
        img_width = 1000
        img_heigth = 2000
        x_max = fxt_normalized_annotation_polygon_dict["shape"]["points"][2]["x"]
        y_max = fxt_normalized_annotation_polygon_dict["shape"]["points"][2]["y"]
        x_min = fxt_normalized_annotation_polygon_dict["shape"]["points"][0]["x"]
        y_min = fxt_normalized_annotation_polygon_dict["shape"]["points"][0]["y"]

        # Act
        annotation = NormalizedAnnotationRESTConverter.normalized_annotation_from_dict(
            fxt_normalized_annotation_polygon_dict,
            image_width=img_width,
            image_height=img_heigth,
        )

        # Assert
        assert annotation.shape.type == ShapeType.POLYGON
        assert annotation.shape.x_max == int(img_width * x_max)
        assert annotation.shape.y_max == int(img_heigth * y_max)
        assert annotation.shape.area == (
            (int(img_width * x_max) - int(img_width * x_min)) * (int(img_heigth * y_max) - int(img_heigth * y_min))
        )

    def test_to_normalized_dict(self, fxt_annotation_scene_from_normalized: AnnotationScene):
        # Act
        annotation_rest = NormalizedAnnotationRESTConverter.to_normalized_dict(
            fxt_annotation_scene_from_normalized, image_height=2000, image_width=1000
        )

        # Assert
        rect_rest = annotation_rest["annotations"][0]
        polygon_rest = annotation_rest["annotations"][1]
        assert rect_rest["shape"]["type"] == "RECTANGLE"
        assert polygon_rest["shape"]["type"] == "POLYGON"
        assert rect_rest["shape"]["x"] == 0.05
        assert rect_rest["shape"]["width"] == 0.9
        assert polygon_rest["shape"]["points"][0] == {"x": 0.1, "y": 0.2}

    def test_normalized_annotation_scene_from_dict(
        self,
        fxt_normalized_annotation_polygon_dict,
        fxt_normalized_annotation_dict,
        fxt_image_identifier_rest,
    ):
        # Arrange
        img_width = 1000
        img_height = 2000
        annotation_scene_dict = {
            "annotations": [
                fxt_normalized_annotation_dict,
                fxt_normalized_annotation_polygon_dict,
            ],
            "kind": "annotation",
            "media_identifier": fxt_image_identifier_rest,
        }

        # Act
        annotation_scene = NormalizedAnnotationRESTConverter.normalized_annotation_scene_from_dict(
            annotation_scene_dict, image_width=img_width, image_height=img_height
        )

        # Assert
        assert annotation_scene.kind == AnnotationKind.ANNOTATION
        assert annotation_scene.annotations[0].shape.type == ShapeType.RECTANGLE
        assert annotation_scene.annotations[1].shape.type == ShapeType.POLYGON
