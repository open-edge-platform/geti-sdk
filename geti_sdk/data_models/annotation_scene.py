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
import logging
from collections.abc import Sequence
from pprint import pformat
from typing import Any, ClassVar

import attr
import cv2
import numpy as np

from geti_sdk.data_models.annotations import Annotation
from geti_sdk.data_models.enums import AnnotationKind
from geti_sdk.data_models.label import Label, ScoredLabel
from geti_sdk.data_models.media import MediaInformation
from geti_sdk.data_models.media_identifiers import ImageIdentifier, VideoFrameIdentifier
from geti_sdk.data_models.shapes import (
    Ellipse,
    Keypoint,
    Polygon,
    Rectangle,
    RotatedRectangle,
    Shape,
)
from geti_sdk.data_models.task_annotation_state import TaskAnnotationState
from geti_sdk.data_models.utils import (
    attr_value_serializer,
    deidentify,
    remove_null_fields,
    round_dictionary,
    str_to_annotation_kind,
    str_to_datetime,
)


@attr.define
class AnnotationScene:
    """
    Representation of an annotation scene for a certain media entity in GETi. An
    annotation scene holds all annotations for that specific media entity.

    :var annotations: List of
        :py:class:`~geti_sdk.data_models.annotations.Annotation`s belonging to the
        media entity
    :var id: unique database ID of the AnnotationScene in GETi
    :var kind: :py:class:`~geti_sdk.data_models.enums.annotation_kind.AnnotationKind`
        of the annotation (Annotation or Prediction)
    :var media_identifier: Identifier of the media entity to which this AnnotationScene
        applies
    :var modified: Date and time at which this AnnotationScene was last modified
    :var labels_to_revisit_full_scene: Optional list of database ID's of the labels
        that may not be up to date and therefore need to be revisited for this
        AnnotationScene.
    :var annotation_state_per_task: Optional dictionary holding the annotation state
        for this AnnotationScene for each task in the project pipeline.
    """

    _identifier_fields: ClassVar[list[str]] = ["id", "modified"]
    _GET_only_fields: ClassVar[list[str]] = ["annotation_state_per_task"]

    annotations: list[Annotation]
    kind: str = attr.field(converter=str_to_annotation_kind, default=AnnotationKind.ANNOTATION)
    media_identifier: ImageIdentifier | VideoFrameIdentifier | None = None
    id: str | None = None
    modified: str | None = attr.field(converter=str_to_datetime, default=None)
    labels_to_revisit_full_scene: list[str] | None = None
    annotation_state_per_task: list[TaskAnnotationState] | None = None

    @property
    def has_data(self) -> bool:
        """
        Return True if this AnnotationScene has annotation associated with it.
        """
        return len(self.annotations) > 0

    def deidentify(self) -> None:
        """
        Remove all unique database ID's from the annotationscene and the entities it
        contains.
        """
        deidentify(self)
        self.media_identifier = None
        for annotation in self.annotations:
            annotation.deidentify()

    def prepare_for_post(self) -> None:
        """
        Remove all fields that are not valid for making a POST request to the
        /annotations endpoint.

        :return:
        """
        for field_name in self._GET_only_fields:
            setattr(self, field_name, None)

    def append(self, annotation: Annotation) -> None:
        """
        Add an annotation to the annotation scene.

        :param annotation: Annotation to add
        """
        self.annotations.append(annotation)

    def get_by_shape(self, shape: Shape) -> Annotation | None:
        """
        Return the annotation belonging to a specific shape. Returns None if no
        Annotation is found for the shape.

        :param shape: Shape to return the annotation for
        :return:
        """
        return next(
            (annotation for annotation in self.annotations if annotation.shape == shape),
            None,
        )

    def extend(self, annotations: list[Annotation]):
        """
        Extend the list of annotations in the AnnotationScene with additional entries
        in the `annotations` list.

        :param annotations: List of Annotations to add to the AnnotationScene
        :return:
        """
        for annotation in annotations:
            current_shape_annotation = self.get_by_shape(annotation.shape)
            if current_shape_annotation is not None:
                current_shape_annotation.extend_labels(annotation.labels)
            else:
                self.annotations.append(annotation)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the AnnotationScene to a dictionary representation.

        :return: Dictionary holding the annotation scene data
        """
        return attr.asdict(self, value_serializer=attr_value_serializer)

    @property
    def overview(self) -> str:
        """
        Return a string that gives an overview of the annotation scene.

        :return: overview string of the annotation scene
        """
        deidentified = copy.deepcopy(self)
        deidentified.deidentify()
        dict_output = deidentified.to_dict()
        for annotation_dict in dict_output["annotations"]:
            shape_dict = annotation_dict["shape"]
            annotation_dict["shape"] = round_dictionary(shape_dict)
        remove_null_fields(dict_output)
        return pformat(dict_output)

    @staticmethod
    def _add_shape_to_mask(
        shape: Shape,
        mask: np.ndarray,
        labels: list[ScoredLabel],
        color: tuple[int, int, int],
        line_thickness: int,
    ) -> np.ndarray:
        """
        Draw an GETi shape entity `shape` on the pixel level mask `mask`. The shape
        will be drawn in the color specified as R,G,B tuple in `color`, using a line
        thickness `line_thickness` (in pixels).

        If the shape represents classification annotations or predictions, this method
        will print the name of the classification labels in the lower-left corner of
        the mask

        :param shape: Shape to add
        :param mask: Mask to draw the shape on
        :param labels: List of labels belonging to the shape
        :param color: RGB tuple representing the color in which the shape should be
            drawn
        :param line_thickness: Line thickness (in pixels) with which the shape should
            be drawn
        :return: Mask with the shape drawn on it
        """
        image_height, image_width = mask.shape[0:-1]
        if isinstance(shape, Ellipse | Rectangle):
            x, y = int(shape.x), int(shape.y)
            width, height = int(shape.width), int(shape.height)
            if isinstance(shape, Ellipse):
                cv2.ellipse(
                    mask,
                    center=(x, y),
                    axes=(width, height),
                    angle=0,
                    color=color,
                    startAngle=0,
                    endAngle=360,
                    thickness=line_thickness,
                )
                cv2.ellipse(
                    mask,
                    center=(x, y),
                    axes=(width, height),
                    angle=0,
                    color=(1, 1, 1),
                    startAngle=0,
                    endAngle=360,
                    thickness=1,
                )
            elif isinstance(shape, Rectangle):
                if not shape.is_full_box(image_width=image_width, image_height=image_height):
                    cv2.rectangle(
                        mask,
                        pt1=(x, y),
                        pt2=(x + width, y + height),
                        color=color,
                        thickness=line_thickness,
                    )
                    cv2.rectangle(
                        mask,
                        pt1=(x, y),
                        pt2=(x + width, y + height),
                        color=(1, 1, 1),
                        thickness=1,
                    )
                else:
                    origin = [int(0.01 * image_width), int(0.99 * image_height)]
                    for label in labels:
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 1
                        cv2.putText(
                            mask,
                            label.name,
                            org=origin,
                            fontFace=font,
                            fontScale=font_scale,
                            color=label.color_tuple,
                            thickness=1,
                        )
                        text_width, text_height = cv2.getTextSize(label.name, font, font_scale, line_thickness)[0]
                        origin[0] += text_width + 2
        elif isinstance(shape, Keypoint):
            x, y = int(shape.x), int(shape.y)
            cv2.circle(
                mask,
                center=(x, y),
                radius=5,
                color=color,
                thickness=line_thickness,
            )
            cv2.circle(
                mask,
                center=(x, y),
                radius=5,
                color=(1, 1, 1),
                thickness=1,
            )
        elif isinstance(shape, RotatedRectangle):
            shape = shape.to_polygon()
        if isinstance(shape, Polygon):
            contour = shape.points_as_contour()
            cv2.drawContours(
                mask,
                contours=[contour],
                color=color,
                thickness=line_thickness,
                contourIdx=-1,
            )
            cv2.drawContours(mask, contours=[contour], color=(1, 1, 1), thickness=1, contourIdx=-1)
        return mask

    def as_mask(self, media_information: MediaInformation) -> np.ndarray:
        """
        Convert the shapes in the annotation scene to a mask that can be overlayed on
        an image.

        :param media_information: MediaInformation object containing the width and
            height of the image for which the mask should be generated.
        :return: np.ndarray holding the mask representation of the annotation scene
        """
        image_width = media_information.width
        image_height = media_information.height
        mask = np.zeros((image_height, image_width, 3), dtype=np.uint8)

        for annotation in self.annotations:
            label_to_copy_color = annotation.labels[0]
            line_thickness = 3
            shape = annotation.shape
            mask = self._add_shape_to_mask(
                shape=shape,
                mask=mask,
                labels=annotation.labels,
                color=label_to_copy_color.color_tuple,
                line_thickness=line_thickness,
            )
        return mask

    def get_labels(self) -> list[Label]:
        """
        Return a list of all labels present in the annotation scene.

        :return: List of labels
        """
        labels = set()
        for annotation in self.annotations:
            labels.update(annotation.labels)
        return list(labels)

    def get_label_names(self) -> list[str]:
        """
        Return a list with the unique label names in the annotation scene.

        :return: List of label names
        """
        label_names: set[str] = set()
        for label in self.get_labels():
            label_names.update([label.name])
        return list(label_names)

    def apply_identifier(self, media_identifier: ImageIdentifier | VideoFrameIdentifier) -> "AnnotationScene":
        """
        Apply a `media_identifier` to the current AnnotationScene instance, such
        that the GETi cluster will recognize this AnnotationScene as belonging to the
        media item identified by the media_identifier.

        This method creates and returns a new AnnotationScene instance. The instance
        on which this method is called remains unmodified.

        :param media_identifier: Image or VideoFrame identifier to apply
        :return: new AnnotationScene instance with the identifiers set according to
            `media_identifier`
        """
        new_annotation = copy.deepcopy(self)
        new_annotation.media_identifier = media_identifier
        new_annotation.id = ""
        new_annotation.modified = None
        for annotation in new_annotation.annotations:
            annotation.id = ""
            annotation.modified = ""
        return new_annotation

    def map_labels(self, labels: Sequence[Label | ScoredLabel]) -> "AnnotationScene":
        """
        Attempt to map the labels found in `labels` to those in the AnnotationScene
        instance. Labels are matched by name. This method will return a new
        AnnotationScene object.

        :param labels: Labels to which the existing labels should be mapped
        :return: AnnotationScene with updated labels, corresponding to those found in
            the `project` (if matching labels were found)
        """
        annotations: list[Annotation] = []
        for annotation in self.annotations:
            annotations.append(annotation.map_labels(labels=labels))
        return AnnotationScene(
            annotations=annotations,
            media_identifier=self.media_identifier,
            modified=self.modified,
        )

    def filter_annotations(self, labels: Sequence[Label | ScoredLabel | str]) -> "AnnotationScene":
        """
        Filter annotations in the scene to only include labels that are present in the
        provided list of labels.

        :param labels: List of labels or label names to filter the scene with
        :return: AnnotationScene with filtered annotations
        """
        label_names_to_keep = {label if type(label) is str else label.name for label in labels}
        filtered_annotations: list[Annotation] = []
        for annotation in self.annotations:
            for label_name in annotation.label_names:
                if label_name in label_names_to_keep:
                    filtered_annotations.append(annotation)
                    break
        return AnnotationScene(
            annotations=filtered_annotations,
            media_identifier=self.media_identifier,
            modified=self.modified,
        )

    def resolve_label_names_and_colors(self, labels: list[Label]) -> None:
        """
        Add label names and colors to all annotations, based on a list of available
        labels.

        :param labels: List of labels for the project, serving as a reference point
            for label names and colors
        """
        name_map = {label.id: label.name for label in labels}
        color_map = {label.id: label.color for label in labels}
        for annotation in self.annotations:
            for label in annotation.labels:
                label.name = name_map.get(label.id, None)
                label.color = color_map.get(label.id, None)
                if label.name is None:
                    logging.warning(f"Unable to resolve label details for label with id `{label.id}`")
