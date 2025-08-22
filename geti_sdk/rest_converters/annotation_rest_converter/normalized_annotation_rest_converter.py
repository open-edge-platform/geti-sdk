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
from typing import Any

import attr

from geti_sdk.data_models import Annotation, AnnotationScene
from geti_sdk.data_models.enums import ShapeType
from geti_sdk.data_models.label import ScoredLabel
from geti_sdk.data_models.shapes import Shape
from geti_sdk.data_models.utils import (
    attr_value_serializer,
    remove_null_fields,
    str_to_shape_type,
)

from .annotation_rest_converter import AnnotationRESTConverter


class NormalizedAnnotationRESTConverter(AnnotationRESTConverter):
    """
    Class containing methods for converting annotations in normalized format to
    and from their REST representation

    It is a legacy class to support the annotation format in a normalized coordinate
    system, which was used in SCv1.1 and below
    """

    @staticmethod
    def _normalized_shape_from_dict(input_dict: dict[str, Any], image_width: int, image_height: int) -> Shape:
        """
        Legacy method to convert shapes represented in normalized coordinates to Shape
        objects. This method is used for reading annotations in SCv1.1 or lower format.

        :param input_dict: Dictionary containing the shape in normalized coordinates
        :param image_width: Width of the image to which the shape applies
        :param image_height: Height of the image to which the shape applies
        :return: Shape object corresponding to the input_dict
        """
        coordinate_keys_x = ["x", "width"]
        coordinate_keys_y = ["y", "height"]
        input_copy = copy.deepcopy(input_dict)
        type_ = str_to_shape_type(input_copy.get("type"))
        if type_ != ShapeType.POLYGON:
            denormalized_coordinates: dict[str, float] = {}
            for key, value in input_copy.items():
                if key in coordinate_keys_x:
                    new_value = value * image_width
                elif key in coordinate_keys_y:
                    new_value = value * image_height
                else:
                    continue
                denormalized_coordinates.update({key: new_value})
            input_copy.update(denormalized_coordinates)
        else:
            points_dicts = input_copy.pop("points")
            points = [{"x": point["x"] * image_width, "y": point["y"] * image_height} for point in points_dicts]
            input_copy.update({"points": points})
        return AnnotationRESTConverter._shape_from_dict(input_dict=input_copy)

    @staticmethod
    def normalized_annotation_from_dict(input_dict: dict[str, Any], image_width: int, image_height: int) -> Annotation:
        """
        Legacy method that converts a dictionary representing an annotation (in
        normalized coordinates) to an Annotation object

        :param input_dict:
        :param image_width: Width of the image to which the annotation applies
        :param image_height: Height of the image to which the annotation applies
        :return: Annotation object corresponding to input_dict
        """
        input_copy = copy.deepcopy(input_dict)
        labels: list[ScoredLabel] = []
        for label in input_dict["labels"]:
            labels.append(AnnotationRESTConverter._scored_label_from_dict(label))
        shape = NormalizedAnnotationRESTConverter._normalized_shape_from_dict(
            input_dict["shape"], image_width=image_width, image_height=image_height
        )
        input_copy.update({"labels": labels, "shape": shape})
        return Annotation(**input_copy)

    @staticmethod
    def normalized_annotation_scene_from_dict(
        annotation_scene: dict[str, Any], image_width: int, image_height: int
    ) -> AnnotationScene:
        """
        Legacy method that creates an AnnotationScene object from a dictionary
        returned by the /annotations REST endpoint in GETi versions 1.1 or below

        :param annotation_scene: dictionary representing an AnnotationScene, which
            contains all annotations for a certain media entity
        :param image_width: Width of the image to which the annotation scene applies
        :param image_height: Height of the image to which the annotation scene applies
        :return: AnnotationScene object
        """
        input_copy = copy.deepcopy(annotation_scene)
        annotations: list[Annotation] = []
        for annotation in annotation_scene["annotations"]:
            annotations.append(
                NormalizedAnnotationRESTConverter.normalized_annotation_from_dict(
                    input_dict=annotation,
                    image_width=image_width,
                    image_height=image_height,
                )
            )
        input_copy.update({"annotations": annotations})
        return AnnotationRESTConverter.from_dict(input_copy)

    @staticmethod
    def to_normalized_dict(
        annotation_scene: AnnotationScene,
        image_width: int,
        image_height: int,
        deidentify: bool = True,
    ) -> dict[str, Any]:
        """
        Convert an AnnotationScene to a dictionary. By default, removes any ID
        fields in the output dictionary

        :param annotation_scene: AnnotationScene object to convert
        :param image_width:
        :param image_height:
        :param deidentify: True to remove any unique database ID fields in the output,
            False to keep these fields. Defaults to True
        :return: Dictionary holding the serialized AnnotationScene data
        """
        if deidentify:
            annotation_scene.deidentify()
        annotation_scene_dict = attr.asdict(annotation_scene, recurse=True, value_serializer=attr_value_serializer)
        annotations_serialized: list[dict[str, Any]] = []
        for annotation in annotation_scene.annotations:
            annotation_dict = attr.asdict(annotation, recurse=True, value_serializer=attr_value_serializer)
            annotation_dict["shape"] = annotation.shape.to_normalized_coordinates(
                image_width=image_width, image_height=image_height
            )
            annotations_serialized.append(annotation_dict)
        annotation_scene_dict["annotations"] = annotations_serialized
        remove_null_fields(annotation_scene_dict)
        return annotation_scene_dict
