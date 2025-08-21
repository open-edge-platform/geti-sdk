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
from typing import Any, cast

import attr
from omegaconf import OmegaConf

from geti_sdk.data_models import Annotation, AnnotationScene
from geti_sdk.data_models.enums import ShapeType
from geti_sdk.data_models.label import ScoredLabel
from geti_sdk.data_models.media import MediaType
from geti_sdk.data_models.media_identifiers import (
    ImageIdentifier,
    MediaIdentifier,
    VideoFrameIdentifier,
)
from geti_sdk.data_models.shapes import (
    Ellipse,
    Keypoint,
    Point,
    Polygon,
    Rectangle,
    RotatedRectangle,
    Shape,
)
from geti_sdk.data_models.task_annotation_state import TaskAnnotationState
from geti_sdk.data_models.utils import (
    attr_value_serializer,
    remove_null_fields,
    str_to_media_type,
    str_to_shape_type,
)

SHAPE_TYPE_MAPPING = {
    ShapeType.ELLIPSE: Ellipse,
    ShapeType.RECTANGLE: Rectangle,
    ShapeType.POLYGON: Polygon,
    ShapeType.ROTATED_RECTANGLE: RotatedRectangle,
    ShapeType.KEYPOINT: Keypoint,
}
MEDIA_IDENTIFIER_MAPPING = {
    MediaType.IMAGE: ImageIdentifier,
    MediaType.VIDEO_FRAME: VideoFrameIdentifier,
}


class AnnotationRESTConverter:
    """
    Class to convert REST representations of annotations into AnnotationScene entities.
    """

    @staticmethod
    def to_dict(annotation_scene: AnnotationScene, deidentify: bool = True) -> dict[str, Any]:
        """
        Convert an AnnotationScene to a dictionary. By default, removes any ID
        fields in the output dictionary

        :param annotation_scene: AnnotationScene object to convert
        :param deidentify: True to remove any unique database ID fields in the output,
            False to keep these fields. Defaults to True
        :return: Dictionary holding the serialized AnnotationScene data
        """
        if deidentify:
            annotation_scene.deidentify()
        annotation_dict = attr.asdict(annotation_scene, recurse=True, value_serializer=attr_value_serializer)
        remove_null_fields(annotation_dict)
        return annotation_dict

    @staticmethod
    def _shape_from_dict(input_dict: dict[str, Any]) -> Shape:
        """
        Convert a dictionary representing a shape to a Shape object.

        :param input_dict:
        :return: Shape corresponding to the input dict
        """
        input_copy = copy.deepcopy(input_dict)
        type_ = str_to_shape_type(input_copy.get("type"))
        class_type = SHAPE_TYPE_MAPPING[type_]
        if issubclass(class_type, Polygon):
            points_dicts = input_copy.pop("points")
            points = [Point(**point) for point in points_dicts]
            input_copy.update({"points": points})
        return class_type(**input_copy)

    @staticmethod
    def _scored_label_from_dict(input_dict: dict[str, Any]) -> ScoredLabel:
        """
        Create a ScoredLabel object from an input dictionary.

        :param input_dict:
        :return:
        """
        label_dict_config = OmegaConf.create(input_dict)
        schema = OmegaConf.structured(ScoredLabel)
        values = OmegaConf.merge(schema, label_dict_config)
        return cast("ScoredLabel", OmegaConf.to_object(values))

    @staticmethod
    def annotation_from_dict(input_dict: dict[str, Any]) -> Annotation:
        """
        Convert a dictionary representing an annotation to an Annotation object.

        :param input_dict:
        :return:
        """
        input_copy = copy.deepcopy(input_dict)
        labels: list[ScoredLabel] = []
        for label in input_dict["labels"]:
            labels.append(AnnotationRESTConverter._scored_label_from_dict(label))
        shape = AnnotationRESTConverter._shape_from_dict(input_dict["shape"])
        input_copy.update({"labels": labels, "shape": shape})
        return Annotation(**input_copy)

    @staticmethod
    def _media_identifier_from_dict(input_dict: dict[str, Any]) -> MediaIdentifier:
        """
        Convert a dictionary representing a media identifier to a MediaIdentifier
        object.

        :param input_dict:
        :return:
        """
        if isinstance(input_dict, MediaIdentifier):
            return input_dict
        type_ = str_to_media_type(input_dict["type"])
        identifier_type = MEDIA_IDENTIFIER_MAPPING[type_]
        return identifier_type(**input_dict)

    @staticmethod
    def from_dict(annotation_scene: dict[str, Any]) -> AnnotationScene:
        """
        Create an AnnotationScene object from a dictionary returned by the
        /annotations REST endpoint in the Intel® Geti™ platform.

        :param annotation_scene: dictionary representing an AnnotationScene, which
            contains all annotations for a certain media entity
        :return: AnnotationScene object
        """
        input_copy = copy.deepcopy(annotation_scene)
        annotations: list[Annotation] = []
        for annotation in annotation_scene["annotations"]:
            if not isinstance(annotation, Annotation):
                annotations.append(AnnotationRESTConverter.annotation_from_dict(annotation))
            else:
                annotations.append(annotation)
        media_identifier = AnnotationRESTConverter._media_identifier_from_dict(annotation_scene["media_identifier"])
        input_copy.update({"annotations": annotations, "media_identifier": media_identifier})
        if "annotation_state_per_task" in annotation_scene:
            annotation_states: list[TaskAnnotationState] = []
            for annotation_state in annotation_scene["annotation_state_per_task"]:
                annotation_states.append(TaskAnnotationState(**annotation_state))
            input_copy.update({"annotation_state_per_task": annotation_states})
        return AnnotationScene(**input_copy)
