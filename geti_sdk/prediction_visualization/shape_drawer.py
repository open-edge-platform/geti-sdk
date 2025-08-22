# Copyright (C) 2024 Intel Corporation
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

"""The module implements helpers for drawing shapes."""

# pylint: disable=too-many-instance-attributes, too-many-arguments, too-many-locals
import abc
from collections.abc import Callable, Sequence
from typing import (
    Generic,
    NewType,
    TypeVar,
)

import cv2
import numpy as np

from geti_sdk.data_models.annotation_scene import AnnotationScene
from geti_sdk.data_models.annotations import Annotation
from geti_sdk.data_models.label import Label, ScoredLabel
from geti_sdk.data_models.predictions import Prediction
from geti_sdk.data_models.shapes import (
    Ellipse,
    Keypoint,
    Point,
    Polygon,
    Rectangle,
    RotatedRectangle,
    Shape,
)

CvTextSize = NewType("CvTextSize", tuple[tuple[int, int], int])

_Any = TypeVar("_Any")


class DrawerEntity(Generic[_Any]):
    """An interface to draw a shape of type ``T`` onto an image."""

    supported_types: Sequence[type[Shape]] = []

    @abc.abstractmethod
    def draw(self, image: np.ndarray, entity: _Any, labels: list[ScoredLabel]) -> np.ndarray:
        """
        Draw an entity to a given frame.

        :param image: The image to draw the entity on.
        :param entity: The entity to draw.
        :param labels: Labels of the shapes to draw
        :return: frame with shape drawn on it
        """
        raise NotImplementedError


class Helpers:
    """
    Contains variables which are used by all subclasses.

    Contains functions which help with generating coordinates, text and text scale.
    These functions are use by the DrawerEntity Classes when drawing to an image.
    """

    def __init__(self) -> None:
        # Same alpha value that the UI uses for Labels
        self.alpha_shape = 100 / 256
        self.alpha_labels = 153 / 256
        self.assumed_image_width_for_text_scale = 1500  # constant number for size of classification/counting overlay
        self.top_margin = 0.07  # part of the top screen reserved for top left classification/counting overlay
        self.content_padding = 3
        self.top_left_box_thickness = 1
        self.content_margin = 2
        self.label_offset_box_shape = 10
        self.black = (0, 0, 0)
        self.white = (255, 255, 255)
        self.yellow = (255, 255, 0)

        self.cursor_pos = Point(0, 0)
        self.line_height = 0
        self.font = cv2.FONT_HERSHEY_DUPLEX

    @staticmethod
    def draw_transparent_rectangle(
        img: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        color: tuple[int, int, int],
        alpha: float,
    ) -> np.ndarray:
        """
        Draw a rectangle on an image.

        :param img: Image
        :param x1: Left side
        :param y1: Top side
        :param x2: Right side
        :param y2: Bottom side
        :param color: Color
        :param alpha: Alpha value between 0 and 1
        :return: Image with rectangle drawn on it
        """
        x1 = np.clip(x1, 0, img.shape[1] - 1)
        y1 = np.clip(y1, 0, img.shape[0] - 1)
        x2 = np.clip(x2 + 1, 0, img.shape[1] - 1)
        y2 = np.clip(y2 + 1, 0, img.shape[0] - 1)
        rect = img[y1:y2, x1:x2]
        rect[:] = (alpha * np.array(color))[np.newaxis, np.newaxis] + (1 - alpha) * rect
        return img

    def generate_text_scale(self, image: np.ndarray) -> float:
        """
        Calculate the scale of the text.

        :param image: Image to calculate the text scale for.
        :return: Scale for the text
        """
        return round(image.shape[1] / self.assumed_image_width_for_text_scale, 1)

    @staticmethod
    def generate_text_for_label(label: Label | ScoredLabel, show_labels: bool, show_confidence: bool) -> str:
        """
        Return a string representing a given label and its associated probability if label is a ScoredLabel.

        :param label: Label
        :param show_labels: Whether to render the labels above the shape
        :param show_confidence: Whether to render the confidence above the shape
        :return: Formatted string (e.g. `"Cat 58%"`)
        """
        text = ""
        if show_labels:
            text += label.name
        if show_confidence and isinstance(label, ScoredLabel):
            if len(text) > 0:
                text += " "
            text += f"{label.probability:.0%}"
        return text

    def generate_draw_command_for_labels(
        self,
        labels: Sequence[Label | ScoredLabel],
        image: np.ndarray,
        show_labels: bool,
        show_confidence: bool,
    ) -> tuple[Callable[[np.ndarray], np.ndarray], int, int]:
        """
        Generate draw function and content width and height for labels.

        Generates a function which can be called to draw a list of labels onto an image relatively to the
        cursor position.
        The width and height of the content is also returned and can be determined to compute
        the best position for content before actually drawing it.

        :param labels: List of labels
        :param image: Image (used to compute font size)
        :param show_labels: Whether to show the label name
        :param show_confidence: Whether to show the confidence probability
        :return: A tuple containing the drawing function, the content width, and the content height
        """
        draw_commands = []
        content_width = 0
        content_height = 0

        # Loop through the list of labels and create a function which can be used to draw the label.
        n_labels = len(labels)
        for i, label in enumerate(labels):
            text = self.generate_text_for_label(label, show_labels, show_confidence)
            if i < n_labels - 1:
                text += " >"
            text_scale = self.generate_text_scale(image)
            thickness = int(text_scale / 2)
            color = label.color_tuple

            item_command, item_width, item_height = self.generate_draw_command_for_text(
                text, text_scale, thickness, color
            )

            draw_commands.append(item_command)

            content_width += item_width
            content_height = max(content_height, item_height)

        def draw_command(img: np.ndarray) -> np.ndarray:
            for command in draw_commands:
                img = command(img)
            return img

        return draw_command, content_width, content_height

    def generate_draw_command_for_text(
        self, text: str, text_scale: float, thickness: int, color: tuple[int, int, int]
    ) -> tuple[Callable[[np.ndarray], np.ndarray], int, int]:
        """
        Generate function to draw text on image relative to cursor position.

        Generate a function which can be called to draw the given text onto an image
        relatively to the cursor position.

        The width and height of the content is also returned and can be determined to compute
        the best position for content before actually drawing it.

        :param text: Text to draw
        :param text_scale: Font size
        :param thickness: Thickness of the text
        :param color: Color of the text
        :return: A tuple containing the drawing function, the content width, and the content height
        """
        padding = self.content_padding
        margin = self.content_margin

        (text_width, text_height), baseline = cv2.getTextSize(
            text, self.font, fontScale=text_scale, thickness=thickness
        )

        width = text_width + 2 * padding
        height = text_height + baseline + 2 * padding
        content_width = width + margin

        text_color = self.black if (color[0] + color[1] + color[2]) / 3 > 200 else self.white

        def draw_command(img: np.ndarray) -> np.ndarray:
            cursor_pos = Point(int(self.cursor_pos.x), int(self.cursor_pos.y))
            self.draw_transparent_rectangle(
                img,
                int(cursor_pos.x),
                int(cursor_pos.y),
                int(cursor_pos.x + width),
                int(cursor_pos.y + height),
                color,
                self.alpha_labels,
            )

            img = cv2.putText(
                img=img,
                text=text,
                org=(
                    cursor_pos.x + padding,
                    cursor_pos.y + height - padding - baseline,
                ),
                fontFace=self.font,
                fontScale=text_scale,
                color=text_color,
                thickness=thickness,
                lineType=cv2.LINE_AA,
            )

            self.cursor_pos.x += content_width
            self.line_height = height

            return img

        return draw_command, content_width, height

    @staticmethod
    def draw_flagpole(
        image: np.ndarray,
        flagpole_start_point: Point,
        flagpole_end_point: Point,
        color: tuple[int, int, int] = (0, 0, 0),
    ):
        """
        Draw a small flagpole between two points.

        :param image: Image
        :param flagpole_start_point: Start of the flagpole
        :param flagpole_end_point: End of the flagpole
        :return: Image
        """
        return cv2.line(
            image,
            flagpole_start_point.as_int_tuple(),
            flagpole_end_point.as_int_tuple(),
            color=color,
            thickness=2,
        )

    def newline(self):
        """Move the cursor to the next line."""
        self.cursor_pos.x = 0
        self.cursor_pos.y += self.line_height + self.content_margin

    def set_cursor_pos(self, cursor_pos: Point | None = None) -> None:
        """
        Move the cursor to a new position.

        :param cursor_pos: New position of the cursor; (0,0) if not specified.
        """
        if cursor_pos is None:
            cursor_pos = Point(0, 0)

        self.cursor_pos = cursor_pos


class ShapeDrawer(DrawerEntity[AnnotationScene]):
    """
    ShapeDrawer to draw any shape on a numpy array. Will overlay the shapes in the same way that the UI does.

    :param show_count: Whether or not to render the amount of objects on screen in the top left.
    :param is_one_label: Whether there is only one label present in the project.
    """

    def __init__(
        self,
        show_count: bool,
        is_one_label: bool,
        show_labels: bool,
        show_confidence: bool,
    ):
        super().__init__()
        self.show_labels = show_labels
        self.show_confidence = show_confidence
        self.show_count = show_count
        self.is_one_label = is_one_label

        if self.is_one_label and not self.show_count:
            self.show_labels = False

        self.shape_drawers = [
            self.RectangleDrawer(self.show_labels, self.show_confidence),
            self.PolygonDrawer(self.show_labels, self.show_confidence),
            self.EllipseDrawer(self.show_labels, self.show_confidence),
            self.KeypointDrawer(self.show_labels, self.show_confidence),
        ]

        # Always show global labels, especially if shape labels are disabled (because of is_one_label).
        self.top_left_drawer = self.TopLeftDrawer(True, self.show_confidence, self.is_one_label)

    def draw(
        self,
        image: np.ndarray,
        scene: AnnotationScene | Prediction,
        labels: list[ScoredLabel],
        fill_shapes: bool = True,
    ) -> np.ndarray:
        """
        Use a compatible drawer to draw all shapes of an annotation to the corresponding image.

        Also render a label in the top left if we need to.

        :param image: Numpy image, one frame of a video on which to draw
        :param scene: AnnotationScene scene corresponding to this particular frame of the video
        :param labels: Can be passed as an empty list since they are already present in annotation_scene
        :return: Modified image.
        """
        num_annotations = 0

        self.top_left_drawer.set_cursor_pos()

        for annotation in scene.annotations:
            if isinstance(annotation.shape, Rectangle) and annotation.shape.is_full_box(image.shape[1], image.shape[0]):
                # If is_one_label is activated, don't draw the labels here
                # because we will draw them again outside the loop.
                if not self.is_one_label:
                    image = self.top_left_drawer.draw(image, annotation, labels=[])
            else:
                num_annotations += 1
                for drawer in self.shape_drawers:
                    if type(annotation.shape) in drawer.supported_types and len(annotation.labels) > 0:
                        image = drawer.draw(
                            image,
                            annotation.shape,
                            labels=annotation.labels,
                            fill_shapes=fill_shapes,
                        )
        if self.is_one_label:
            image = self.top_left_drawer.draw_labels(image, scene.get_labels())
        if self.show_count:
            image = self.top_left_drawer.draw_annotation_count(image, num_annotations)
        return image

    class TopLeftDrawer(Helpers, DrawerEntity[Annotation]):
        """Draws labels in an image's top left corner."""

        def __init__(self, show_labels, show_confidence, is_one_label):
            super().__init__()
            self.show_labels = show_labels
            self.show_confidence = show_confidence
            self.is_one_label = is_one_label

        def draw(self, image: np.ndarray, annotation: Annotation, labels: list[ScoredLabel]) -> np.ndarray:
            """
            Draw the labels of a shape in the image top left corner.

            :param image: Image
            :param annotation: Annotation
            :param labels: (Unused) labels to be drawn on the image
            :return: Image with label on top.
            """
            return self.draw_labels(image, annotation.labels)

        def draw_labels(self, image: np.ndarray, labels: Sequence[Label | ScoredLabel]) -> np.ndarray:
            """
            Draw the labels in the image top left corner.

            :param image: Image
            :param labels: Sequence of labels
            :return: Image with label on top.
            """
            show_confidence = self.show_confidence if not self.is_one_label else False

            draw_command, _, _ = self.generate_draw_command_for_labels(labels, image, self.show_labels, show_confidence)

            image = draw_command(image)

            if len(labels) > 0:
                self.newline()

            return image

        def draw_annotation_count(self, image: np.ndarray, num_annotations: int) -> np.ndarray:
            """
            Draw the number of annotations to the top left corner of the image.

            :param image: Image
            :param num_annotations: Number of annotations
            :return: Image with annotation count on top.
            """
            text = f"Count: {num_annotations}"
            color = self.yellow

            text_scale = self.generate_text_scale(image)
            draw_command, _, _ = self.generate_draw_command_for_text(
                text, text_scale, self.top_left_box_thickness, color
            )
            image = draw_command(image)

            self.newline()

            return image

    class RectangleDrawer(Helpers, DrawerEntity[Rectangle]):
        """Draws rectangles."""

        supported_types = [Rectangle]

        def __init__(self, show_labels, show_confidence):
            super().__init__()
            self.show_labels = show_labels
            self.show_confidence = show_confidence
            self.label_offset_box_shape = 0

        def draw(
            self,
            image: np.ndarray,
            entity: Rectangle,
            labels: list[ScoredLabel],
            fill_shapes: bool = True,
        ) -> np.ndarray:
            """
            Draw a rectangle on the image along with labels.

            :param image: Image to draw on.
            :param entity: Rectangle to draw.
            :param labels: List of labels.
            :param fill_shapes: Whether to fill the shapes with color.
            :return: Image with rectangle drawn on it.
            """
            base_color = labels[0].color_tuple

            # Draw the rectangle on the image
            x1, y1 = int(entity.x), int(entity.y)
            x2, y2 = int(entity.x + entity.width), int(entity.y + entity.height)
            if fill_shapes:
                image = self.draw_transparent_rectangle(image, x1, y1, x2, y2, base_color, self.alpha_shape)
            image = cv2.rectangle(img=image, pt1=(x1, y1), pt2=(x2, y2), color=base_color, thickness=2)

            (
                draw_command,
                content_width,
                content_height,
            ) = self.generate_draw_command_for_labels(labels, image, self.show_labels, self.show_confidence)

            # Generate a command to draw the list of labels
            # and compute the actual size of the list of labels.
            y_coord = y1 - self.label_offset_box_shape - content_height
            x_coord = x1

            # put label inside if it is out of bounds at the top of the shape, and shift label to left if needed
            if y_coord < self.top_margin * image.shape[0]:
                y_coord = y1 + self.label_offset_box_shape
            if x_coord + content_width > image.shape[1]:
                x_coord = x2 - content_width

            # Draw the list of labels.
            self.set_cursor_pos(Point(x_coord, y_coord))
            return draw_command(image)

    class EllipseDrawer(Helpers, DrawerEntity[Ellipse]):
        """Draws ellipses."""

        supported_types = [Ellipse]

        def __init__(self, show_labels, show_confidence):
            super().__init__()
            self.show_labels = show_labels
            self.show_confidence = show_confidence

        def draw(
            self,
            image: np.ndarray,
            entity: Ellipse,
            labels: list[ScoredLabel],
            fill_shapes: bool = True,
        ) -> np.ndarray:
            """
            Draw the ellipse on the image.

            :param image: Image to draw on.
            :param entity: Ellipse to draw.
            :param labels: Labels to draw.
            :param fill_shapes: Whether to fill the shapes with color.
            :return: Image with the ellipse drawn on it.
            """
            base_color = labels[0].color_tuple
            if entity.width > entity.height:
                axes = (
                    int(entity.major_axis * image.shape[1]),
                    int(entity.minor_axis * image.shape[0]),
                )
            else:
                axes = (
                    int(entity.major_axis * image.shape[0]),
                    int(entity.minor_axis * image.shape[1]),
                )
            center = (
                int(entity.x_center * image.shape[1]),
                int(entity.y_center * image.shape[0]),
            )
            # Draw the shape on the image
            alpha = self.alpha_shape
            if fill_shapes:
                overlay = cv2.ellipse(
                    img=image.copy(),
                    center=center,
                    axes=axes,
                    angle=0,
                    startAngle=0,
                    endAngle=360,
                    color=base_color,
                    thickness=cv2.FILLED,
                )
                result_without_border = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
            else:
                result_without_border = image
            result_with_border = cv2.ellipse(
                img=result_without_border,
                center=center,
                axes=axes,
                angle=0,
                startAngle=0,
                endAngle=360,
                color=base_color,
                lineType=cv2.LINE_AA,
            )

            # Generate a command to draw the list of labels
            # and compute the actual size of the list of labels.
            (
                draw_command,
                content_width,
                content_height,
            ) = self.generate_draw_command_for_labels(labels, image, self.show_labels, self.show_confidence)

            # get top left corner of imaginary bbox around circle
            offset = self.label_offset_box_shape
            x_coord = entity.x1 * image.shape[1]
            y_coord = entity.y1 * image.shape[0] - offset - content_height

            flagpole_end_point = Point(entity.get_center_point())

            # put label at bottom if it is out of bounds at the top of the shape, and shift label to left if needed
            if y_coord < self.top_margin * image.shape[0]:
                y_coord = (entity.y1 * image.shape[0]) + (entity.y2 * image.shape[0]) + offset
                flagpole_start_point = Point(x_coord + 1, y_coord)
            else:
                flagpole_start_point = Point(x_coord + 1, y_coord + content_height)

            if x_coord + content_width > result_with_border.shape[1]:
                # The list of labels is too close to the right side of the image.
                # Move it slightly to the left.
                x_coord = result_with_border.shape[1] - content_width

            # Draw the list of labels and a small flagpole.
            self.set_cursor_pos(Point(x_coord, y_coord))
            image = draw_command(result_with_border)
            return self.draw_flagpole(image, flagpole_start_point, flagpole_end_point, labels[0].color_tuple)

    class PolygonDrawer(Helpers, DrawerEntity[Polygon]):
        """Draws polygons."""

        supported_types = [Polygon, RotatedRectangle]

        def __init__(self, show_labels, show_confidence):
            super().__init__()
            self.show_labels = show_labels
            self.show_confidence = show_confidence

        def draw(
            self,
            image: np.ndarray,
            entity: Polygon | RotatedRectangle,
            labels: list[ScoredLabel],
            fill_shapes: bool = True,
        ) -> np.ndarray:
            """
            Draw polygon and labels on image.

            :param image: Image to draw on.
            :param entity: Polygon to draw.
            :param labels: List of labels to draw.
            :param fill_shapes: Whether to fill the shapes with color.
            :return: Image with polygon drawn on it.
            """
            if isinstance(entity, RotatedRectangle):
                entity = entity.to_polygon()
            base_color = labels[0].color_tuple

            # Draw the shape on the image
            alpha = self.alpha_shape
            contours = np.array(
                [[point.x, point.y] for point in entity.points],
                dtype=np.int32,
            )
            if fill_shapes:
                overlay = cv2.drawContours(
                    image=image.copy(),
                    contours=[contours],
                    contourIdx=-1,
                    color=base_color,
                    thickness=cv2.FILLED,
                )
                result_without_border = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
            else:
                result_without_border = image
            result_with_border = cv2.drawContours(
                image=result_without_border,
                contours=[contours],
                contourIdx=-1,
                color=base_color,
                thickness=2,
                lineType=cv2.LINE_AA,
            )

            # Generate a command to draw the list of labels
            # and compute the actual size of the list of labels.
            (
                draw_command,
                content_width,
                content_height,
            ) = self.generate_draw_command_for_labels(labels, image, self.show_labels, self.show_confidence)

            # get point in the center of imaginary bbox around polygon
            x_coords, y_coords = zip(*[(point[0], point[1]) for point in contours])
            x_coord = (2 * np.median(x_coords) + np.min(x_coords)) / 3
            y_coord = min(y_coords) - self.label_offset_box_shape - content_height

            # end point = Y is the median poly Y, x offset to make line flush with text rectangle
            flagpole_end_point = Point(x_coord + 1, np.median(y_coords))

            if y_coord < self.top_margin * image.shape[0]:
                # The polygon is too close to the top of the image.
                # Draw the labels underneath the polygon instead.
                y_coord = max(y_coords) + self.label_offset_box_shape
                flagpole_start_point = Point(x_coord + 1, y_coord)
            else:
                flagpole_start_point = Point(x_coord + 1, y_coord + content_height)

            if x_coord + content_width > result_with_border.shape[1]:
                # The list of labels is too close to the right side of the image.
                # Move it slightly to the left.
                x_coord = result_with_border.shape[1] - content_width

            # Draw the list of labels and a small flagpole.
            self.set_cursor_pos(Point(x_coord, y_coord))
            image = draw_command(result_with_border)
            return self.draw_flagpole(image, flagpole_start_point, flagpole_end_point, labels[0].color_tuple)

    class KeypointDrawer(Helpers, DrawerEntity[Keypoint]):
        """
        Class to draw keypoints on an image.
        """

        supported_types = [Keypoint]

        def __init__(self, show_labels, show_confidence):
            super().__init__()
            self.show_labels = show_labels
            self.show_confidence = show_confidence

        def draw(
            self,
            image: np.ndarray,
            entity: Keypoint,
            labels: list[ScoredLabel],
            fill_shapes: bool = True,
        ) -> np.ndarray:
            """
            Draw the keypoint on the image.

            :param image: Image on which to draw the keypoint
            :param entity: Keypoint to draw.
            :param labels: List of labels.
            :param fill_shapes: Whether to fill the shapes with color.
            :return: Image with the keypoint drawn on it
            """
            smaller_side = min(image.shape[0], image.shape[1])
            radius = max(round(smaller_side * 0.01), 1)

            circle = cv2.circle(
                image,
                center=(int(entity.x), int(entity.y)),
                radius=radius,
                color=(0, 255, 0),
                thickness=(-1 if entity.is_visible else 1),  # Fill the circle for visible, draw a circle for invisible
            )

            # Generate a command to draw the list of labels
            # and compute the actual size of the list of labels.
            (
                draw_command,
                content_width,
                content_height,
            ) = self.generate_draw_command_for_labels(labels, image, self.show_labels, self.show_confidence)

            # Get top edge of keypoint
            offset = self.label_offset_box_shape
            x_coord = entity.x
            y_coord = entity.y - radius - offset

            # Put label at bottom if it is out of bounds at the top of the shape, and shift label to left if needed
            if y_coord < self.top_margin * image.shape[0]:
                y_coord = entity.y + radius + offset

            if x_coord + content_width > circle.shape[1]:
                # The list of labels is too close to the right side of the image.
                # Move it slightly to the left.
                x_coord = circle.shape[1] - content_width

            # Draw the list of labels.
            self.set_cursor_pos(Point(x_coord, y_coord))
            return draw_command(circle)
